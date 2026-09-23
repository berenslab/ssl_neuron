""" GICLMorph: GraphDINO plus cross-modal alignment with 2D projections.

Hao et al. 2026, "GICLMorph: Self-supervised 3D neuronal morphology
representation via graph-image contrastive learning", Expert Systems With
Applications 320, 132059. No official code was released; this is a
reimplementation from the paper.

The model has two branches:

* **IMID** (intra-modal instance discrimination, paper section 3.3) -- a graph
  transformer with adjacency-conditioned attention, trained by cross-view
  self-distillation against an EMA teacher with a centered, sharpened target.
  The paper's graph encoder (7 blocks, 8 heads, dim 32, EMA 0.999, tau_t <
  tau_s) is GraphDINO's, so that branch *is* `ssl_neuron.graphdino.GraphDINO`
  and is reused unchanged. `cmid_weight = 0` therefore recovers GraphDINO
  exactly, which is the paper's "Intra-only" ablation.
* **CMID** (cross-modal instance discrimination, section 3.4) -- a ResNet
  embeds one PCA-guided 2D projection of the same cell (see
  `ssl_neuron/projection.py`), and an InfoNCE loss pulls the student's graph
  embedding of view 1 towards it, with the other cells' embeddings of *both*
  modalities as negatives.

Total loss: L = L_IMID + gamma * L_CMID (eq. 14).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ssl_neuron.graphdino import create_model as create_graphdino

# ImageNet statistics, so that `image_pretrained` weights see inputs in the
# range they were trained on. Harmless for a randomly initialized backbone.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageEncoder(nn.Module):
    """ f_delta (a torchvision ResNet without its classifier) followed by the
    image projection head g_delta (paper eq. 9).

    Takes grayscale images (B x 1 x H x W) with values in [0, 255], uint8 or
    float, and replicates them to three channels so that ImageNet weights can
    be used unchanged.
    """
    def __init__(self, out_dim, backbone='resnet50', pretrained=False, hidden_dim=512):
        super().__init__()
        import torchvision

        weights = 'DEFAULT' if pretrained else None
        net = getattr(torchvision.models, backbone)(weights=weights)
        feat_dim = net.fc.in_features  # 2048 for resnet50, 512 for resnet18
        net.fc = nn.Identity()
        self.backbone = net

        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        self.register_buffer('mean', torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, images):
        x = images.float().div(255.0).expand(-1, 3, -1, -1)
        x = (x - self.mean) / self.std
        return self.head(self.backbone(x))


def cmid_loss(z, h, temperature=0.1):
    """ Symmetric cross-modal InfoNCE of paper eqs. 10-13.

    For anchor z_i the positive is h_i; the denominator sums over every h_K
    (eq. 11) *and* every other z_K of the same modality (eq. 10), so the loss
    also pushes different cells apart within the anchor's modality. The same
    with the roles of z and h swapped, averaged (eq. 13). Similarities are
    cosine (s(.,.) in the paper).

    Args:
        z: graph embeddings (N x D).
        h: image embeddings (N x D), row i belongs to the same cell as z[i].
        temperature: tau.
    """
    z = F.normalize(z, dim=-1)
    h = F.normalize(h, dim=-1)
    n = z.shape[0]
    target = torch.arange(n, device=z.device)
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)

    def one_direction(anchor, other):
        cross = anchor @ other.T / temperature  # n in eq. 12, positive on the diagonal
        intra = (anchor @ anchor.T / temperature).masked_fill(self_mask, float('-inf'))  # m
        return F.cross_entropy(torch.cat([cross, intra], dim=1), target)

    return (one_direction(z, h) + one_direction(h, z)) / 2


class GICLMorph(nn.Module):
    """ GraphDINO (IMID) plus the image branch (CMID).

    Only the graph *student* and the image branch get gradients; the teacher
    is an EMA of the student, as in GraphDINO. The embedding to use downstream
    is the student's CLS embedding (the paper extracts representations from
    the student encoder only), available as `self.student_encoder`, exactly as
    on a `GraphDINO` -- so evaluation code works on either model.

    Args:
        dino: a `GraphDINO`.
        image_encoder: an `ImageEncoder` whose output dim matches the graph
            side (see `cmid_graph_input`).
        cmid_weight: gamma in eq. 14.
        cmid_temp: tau in eq. 12.
        cmid_graph_input: which graph vector enters CMID. 'proj' (the paper,
            Fig. 1 and eq. 10) uses z = g_theta(y), the same projector output
            the DINO loss sees. 'cls' instead maps the CLS embedding y through
            a separate head into a shared space, as CrossPoint does.
        shared_dim: output size of that separate head ('cls' only).
    """
    def __init__(self, dino, image_encoder, cmid_weight=0.4, cmid_temp=0.1,
                 cmid_graph_input='proj', shared_dim=128):
        super().__init__()
        self.dino = dino
        self.image_encoder = image_encoder
        self.cmid_weight = cmid_weight
        self.cmid_temp = cmid_temp
        self.cmid_graph_input = cmid_graph_input

        if cmid_graph_input == 'cls':
            dim = dino.student_encoder.mlp_head[-1].out_features
            self.graph_head = nn.Sequential(
                nn.Linear(dim, 4 * dim),
                nn.GELU(),
                nn.Linear(4 * dim, shared_dim),
            )
        elif cmid_graph_input == 'proj':
            self.graph_head = None
        else:
            raise ValueError(f"cmid_graph_input must be 'proj' or 'cls', got {cmid_graph_input!r}")

    @property
    def student_encoder(self):
        return self.dino.student_encoder

    def update_moving_average(self, decay=None):
        self.dino.update_moving_average(decay=decay)

    def forward(self, node_feat1, node_feat2, adj1, adj2, lapl1, lapl2, images):
        """ Returns the total loss and a dict of its detached parts. """
        imid, (emb1, _), (proj1, _) = self.dino(node_feat1, node_feat2, adj1, adj2,
                                                lapl1, lapl2, return_student=True)
        if not self.cmid_weight:
            return imid, {'imid': imid.detach(), 'cmid': torch.zeros_like(imid).detach()}

        # The paper aligns view 1 (z_i^{t1}); view 2 only enters through IMID.
        z = proj1 if self.graph_head is None else self.graph_head(emb1)
        h = self.image_encoder(images)
        cmid = cmid_loss(z, h, temperature=self.cmid_temp)

        loss = imid + self.cmid_weight * cmid
        return loss, {'imid': imid.detach(), 'cmid': cmid.detach()}


def create_model(config):
    """ GICLMorph from a GraphDINO config plus a `giclmorph` block. """
    cfg = config['giclmorph']
    dino = create_graphdino(config)

    graph_input = cfg.get('cmid_graph_input', 'proj')
    out_dim = config['model']['num_classes'] if graph_input == 'proj' else cfg.get('shared_dim', 128)
    image_encoder = ImageEncoder(out_dim=out_dim,
                                 backbone=cfg.get('image_backbone', 'resnet50'),
                                 pretrained=cfg.get('image_pretrained', False),
                                 hidden_dim=cfg.get('image_head_hidden', 512))

    return GICLMorph(dino, image_encoder,
                     cmid_weight=cfg.get('cmid_weight', 0.4),
                     cmid_temp=cfg.get('cmid_temp', 0.1),
                     cmid_graph_input=graph_input,
                     shared_dim=cfg.get('shared_dim', 128))
