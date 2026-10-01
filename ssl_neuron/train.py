import csv
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from ssl_neuron.utils import AverageMeter, compute_eig_lapl_torch_batch


def set_seed(seed):
    """ Seed python, numpy and torch (CPU and CUDA). Call it *before* building
    the model so that the initialization is reproducible too. """
    if seed is None:
        return
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def embedding_health(emb):
    """ Cheap collapse diagnostics of a batch of embeddings (N x D):

    * `emb_eff_rank`: exp of the entropy of the normalized singular values
      (1 = all cells on a line, D = isotropic). Falls towards 1 under
      dimensional collapse.
    * `emb_top1_var`: share of the variance in the first principal component.
    * `emb_mean_cos`: mean pairwise cosine similarity; -> 1 under complete
      collapse (every cell gets the same embedding).
    """
    emb = emb.double().cpu()
    n = emb.shape[0]
    if n < 2:
        return {}
    centered = emb - emb.mean(dim=0)
    s = torch.linalg.svdvals(centered)
    total = s.sum().clamp_min(1e-12)
    p = s / total
    eff_rank = torch.exp(-(p * torch.log(p + 1e-12)).sum())
    top1 = s[0] ** 2 / (s ** 2).sum().clamp_min(1e-12)
    normed = F.normalize(emb, dim=1)
    mean_cos = ((normed @ normed.T).sum() - n) / (n * (n - 1))
    return {'emb_eff_rank': float(eff_rank), 'emb_top1_var': float(top1),
            'emb_mean_cos': float(mean_cos)}


def _git_info():
    here = Path(__file__).resolve().parent
    try:
        def run(*args):
            return subprocess.run(['git', *args], cwd=here, capture_output=True,
                                  text=True, timeout=10, check=True).stdout.strip()
        return {'commit': run('rev-parse', 'HEAD'), 'branch': run('rev-parse', '--abbrev-ref', 'HEAD'),
                'dirty': bool(run('status', '--porcelain'))}
    except Exception:
        return None


def _hms(seconds):
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h}h{m:02d}m' if h else f'{m}m{s:02d}s'


class Trainer(object):
    """ Trains a `GraphDINO` (or, via `GICLTrainer`, a `GICLMorph`).

    Everything a later analysis needs ends up in `config['trainer']['ckpt_dir']`:

    * `ckpt_<epoch>.pt` -- model `state_dict`, every `save_ckpt_every` epochs
      and after the last one (what `04`/`07` load);
    * `last.pt` -- model, optimizer, iteration and RNG state, every
      `save_last_every` epochs, for `trainer.resume`;
    * `metrics.csv` -- one row per epoch: train and (every `eval_every` epochs)
      val losses, DINO collapse diagnostics, embedding health, lr, gradient
      norm, speed, GPU memory;
    * `config.json`, `run_info.json` -- the config and the environment of the run.

    Optional `trainer` keys: `epochs` (sets `max_iter` from the loader length;
    otherwise `optimizer.max_iter` is used), `eval_every`, `save_last_every`,
    `grad_clip`, `resume`, and the DINO collapse knobs `ema_end` (cosine
    schedule of the teacher EMA decay from `model.move_avg` up to this value)
    and `teacher_temp_warmup` (`{"start": 0.04, "epochs": 30}`). Both default
    to off, which is GraphDINO's original constant schedule.

    The val loader is only used for monitoring. No checkpoint is selected by
    it, so it stays a clean held-out set for `07`.
    """
    def __init__(self, config, model, dataloaders):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dino = getattr(model, 'dino', model)
        self.config = config

        tcfg = config['trainer']
        self.ckpt_dir = tcfg['ckpt_dir']
        self.save_every = tcfg['save_ckpt_every']
        self.save_last_every = tcfg.get('save_last_every', 5)
        self.eval_every = tcfg.get('eval_every', 5)
        self.grad_clip = tcfg.get('grad_clip')
        self.resume = tcfg.get('resume', False)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        ### datasets
        self.train_loader = dataloaders[0]
        self.val_loader= dataloaders[1]

        ### trainings params
        if 'epochs' in tcfg:
            self.max_iter = tcfg['epochs'] * len(self.train_loader)
        else:
            self.max_iter = config['optimizer']['max_iter']
        self.init_lr = config['optimizer']['lr']
        self.exp_decay = config['optimizer']['exp_decay']
        self.lr_warmup = torch.linspace(0., self.init_lr,  steps=(self.max_iter // 50)+1)[1:]
        self.lr_decay = self.max_iter // 5

        ### optional DINO schedules
        self.base_ema = config['model'].get('move_avg', 0.999)
        self.ema_end = tcfg.get('ema_end')
        self.target_teacher_temp = self.dino.teacher_temp
        warmup = tcfg.get('teacher_temp_warmup')
        self.temp_warmup = None
        if warmup:
            self.temp_warmup = (warmup['start'], warmup['epochs'] * len(self.train_loader))

        # `name` and `weight_decay` are optional; the defaults are GraphDINO's
        # original plain Adam without weight decay.
        optimizers = {'adam': optim.Adam, 'adamw': optim.AdamW}
        optimizer_cls = optimizers[config['optimizer'].get('name', 'adam')]
        self.optimizer = optimizer_cls(list(self.model.parameters()), lr=0,
                                       weight_decay=config['optimizer'].get('weight_decay', 0.0))

        self.curr_iter = 0
        self.last_decay = None
        self.history = []
        self.log_k = math.log(config['model']['num_classes'])

    def set_lr(self):
        if self.curr_iter < len(self.lr_warmup):
            lr = self.lr_warmup[self.curr_iter]
        else:
            lr = self.init_lr * self.exp_decay ** ((self.curr_iter - len(self.lr_warmup)) / self.lr_decay)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def _set_dino_schedules(self):
        """ Applies the optional teacher-temperature warmup and EMA schedule for
        the current iteration. Returns the teacher EMA decay (None = the
        model's constant one). """
        if self.temp_warmup is not None:
            start, iters = self.temp_warmup
            frac = min(self.curr_iter / max(iters, 1), 1.0)
            self.dino.teacher_temp = start + (self.target_teacher_temp - start) * frac

        if self.ema_end is None:
            return None
        frac = self.curr_iter / max(self.max_iter, 1)
        return self.ema_end - (self.ema_end - self.base_ema) * (math.cos(math.pi * frac) + 1) / 2

    # ------------------------------------------------------------------ steps

    def _inputs(self, data):
        """ Moves a batch to the device and computes the positional encodings. """
        f1, f2, a1, a2 = [x.float().to(self.device, non_blocking=True) for x in data[:4]]
        l1 = compute_eig_lapl_torch_batch(a1)
        l2 = compute_eig_lapl_torch_batch(a2)
        return f1, f2, a1, a2, l1, l2

    def _forward(self, data):
        """ Returns the loss to backpropagate, a dict of detached scalar loss
        parts for logging, and the student's CLS embedding of view 1. """
        loss, (emb, _), _ = self.model(*self._inputs(data), return_student=True)
        return loss, {'loss': loss.detach()}, emb.detach()

    def train(self):
        start_epoch = self._maybe_resume()
        self._write_run_info(start_epoch)
        self._print_header(start_epoch)

        self.t_start = time.time()
        self.start_iter = self.curr_iter
        epoch = start_epoch
        while self.curr_iter < self.max_iter:
            # Run one epoch.
            self._run_epoch(epoch)
            done = self.curr_iter >= self.max_iter

            # Always keep the final weights, not only the last multiple of
            # `save_ckpt_every`.
            if epoch % self.save_every == 0 or done:
                self._save_checkpoint(epoch)
            if epoch % self.save_last_every == 0 or done:
                self._save_last(epoch)

            epoch += 1

    def _run_epoch(self, epoch):
        t0 = time.time()
        row = {'epoch': epoch}
        row.update(self._train_epoch(epoch))
        train_time = time.time() - t0

        if len(self.val_loader) and (epoch % self.eval_every == 0 or self.curr_iter >= self.max_iter):
            row.update(self._evaluate())

        elapsed = time.time() - self.t_start
        rate = (self.curr_iter - self.start_iter) / max(elapsed, 1e-9)
        row.update({
            'iter': self.curr_iter,
            'lr': float(self.lr),
            'teacher_temp': float(self.dino.teacher_temp),
            'ema_decay': self.last_decay if self.last_decay is not None else self.base_ema,
            'it_per_s': len(self.train_loader) / max(train_time, 1e-9),
            'epoch_time_s': time.time() - t0,
            'gpu_mem_gb': torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0,
        })
        self.history.append(row)
        self._write_history()
        print(self._format(row, eta=(self.max_iter - self.curr_iter) / max(rate, 1e-9)), flush=True)

        if not math.isfinite(row['loss']):
            raise RuntimeError(f'Loss is {row["loss"]} after epoch {epoch}; stopping. '
                               f'The last good state is in {self.ckpt_dir}.')

    def _train_epoch(self, epoch):
        self.model.train()
        meters = defaultdict(AverageMeter)
        params = list(self.model.parameters())
        max_norm = self.grad_clip if self.grad_clip else float('inf')
        for i, data in enumerate(self.train_loader, 0):
            n = data[2].shape[0]

            self.lr = self.set_lr()
            self.last_decay = self._set_dino_schedules()
            self.optimizer.zero_grad(set_to_none=True)

            loss, parts, _ = self._forward(data)

            # optimize; with no `grad_clip` this only measures the norm
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
            self.optimizer.step()

            # update teacher weights
            self.model.update_moving_average(decay=self.last_decay)

            # Kept as tensors: no GPU sync until the epoch ends.
            for k, v in {**parts, **self.dino.last_stats, 'grad_norm': grad_norm.detach()}.items():
                meters[k].update(v, n)
            self.curr_iter += 1

        return {k: float(m.avg) for k, m in meters.items()}

    @torch.no_grad()
    def _evaluate(self):
        """ Losses and embedding health on the val loader. The val loader's
        generator is reseeded, so every call sees the same augmentations and
        the curve is comparable across epochs. """
        self.model.eval()
        if self.val_loader.generator is not None:
            self.val_loader.generator.manual_seed(0)

        meters = defaultdict(AverageMeter)
        embeddings = []
        for data in self.val_loader:
            n = data[2].shape[0]
            _, parts, emb = self._forward(data)
            for k, v in {**parts, **self.dino.last_stats}.items():
                meters[k].update(v, n)
            embeddings.append(emb.float())
        self.model.train()

        out = {f'val_{k}': float(m.avg) for k, m in meters.items()}
        out.update(embedding_health(torch.cat(embeddings)))
        return out

    # ---------------------------------------------------------------- logging

    def _format(self, row, eta):
        parts = [f'Epoch {row["epoch"]:>4}', f'it {row["iter"]}/{self.max_iter}',
                 f'lr {row["lr"]:.2e}', f'loss {row["loss"]:.4f}']
        if 'imid' in row:
            parts.append(f'IMID {row["imid"]:.4f} CMID {row["cmid"]:.4f}')
        parts.append(f'KL {row["kl"]:.3f}')
        parts.append(f'H_t {row["teacher_entropy"]:.2f}/{self.log_k:.2f} '
                     f'H_s {row["student_entropy"]:.2f} H_marg {row["marginal_entropy"]:.2f}')
        parts.append(f'|g| {row["grad_norm"]:.2f}')
        if 'val_loss' in row:
            parts.append(f'val {row["val_loss"]:.4f}')
            if 'emb_eff_rank' in row:
                parts.append(f'rank {row["emb_eff_rank"]:.1f} cos {row["emb_mean_cos"]:.2f}')
        parts.append(f'{row["it_per_s"]:.1f} it/s')
        parts.append(f'ETA {_hms(eta)}')
        if row['gpu_mem_gb']:
            parts.append(f'{row["gpu_mem_gb"]:.1f}G')
        return ' | '.join(parts)

    def _write_history(self):
        fields = []
        for row in self.history:
            fields += [k for k in row if k not in fields]
        path = os.path.join(self.ckpt_dir, 'metrics.csv')
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.history)

    def _read_history(self, before_epoch):
        path = os.path.join(self.ckpt_dir, 'metrics.csv')
        if not os.path.exists(path):
            return []
        with open(path, newline='') as f:
            rows = [{k: float(v) for k, v in row.items() if v != ''} for row in csv.DictReader(f)]
        for row in rows:
            row['epoch'], row['iter'] = int(row['epoch']), int(row['iter'])
        return [row for row in rows if row['epoch'] < before_epoch]

    def _print_header(self, start_epoch):
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        epochs = math.ceil(self.max_iter / len(self.train_loader))
        print(f'{len(self.train_loader.dataset)} train / {len(self.val_loader.dataset)} val cells | '
              f'{len(self.train_loader)} it/epoch | {epochs} epochs = {self.max_iter} it | '
              f'{n_params / 1e6:.2f}M trainable params | device {self.device}')
        print(f'lr {self.init_lr:g}: warmup {len(self.lr_warmup)} it, then x{self.exp_decay:g} every '
              f'{self.lr_decay} it | grad_clip {self.grad_clip} | eval every {self.eval_every} epochs | '
              f'ckpts in {self.ckpt_dir}')
        print(f'DINO collapse reference: ln(num_classes) = {self.log_k:.2f}. '
              f'H_t ~ ln K: uniform collapse; H_marg ~ 0: one prototype for every cell.')
        if start_epoch:
            print(f'Resumed at epoch {start_epoch}, iteration {self.curr_iter}.')

    def _write_run_info(self, start_epoch):
        with open(os.path.join(self.ckpt_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)

        path = os.path.join(self.ckpt_dir, 'run_info.json')
        info = {}
        if start_epoch and os.path.exists(path):
            with open(path) as f:
                info = json.load(f)
        info.setdefault('resumed', [])
        if start_epoch:
            info['resumed'].append({'epoch': start_epoch, 'time': datetime.now().isoformat(timespec='seconds'),
                                    'git': _git_info()})
        else:
            info.update({
                'started': datetime.now().isoformat(timespec='seconds'),
                'command': sys.argv,
                'host': platform.node(),
                'git': _git_info(),
                'seed': self.config['trainer'].get('seed'),
                'torch': torch.__version__,
                'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'n_train': len(self.train_loader.dataset),
                'n_val': len(self.val_loader.dataset),
                'iters_per_epoch': len(self.train_loader),
                'max_iter': self.max_iter,
                'ln_num_classes': self.log_k,
                'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            })
        with open(path, 'w') as f:
            json.dump(info, f, indent=4)

    # ------------------------------------------------------------ checkpoints

    def _save_checkpoint(self, epoch):
        filename = 'ckpt_{}.pt'.format(epoch)
        PATH = os.path.join(self.ckpt_dir, filename)
        torch.save(self.model.state_dict(), PATH)
        print('Save model after epoch {} as {}.'.format(epoch, filename), flush=True)

    def _save_last(self, epoch):
        """ Everything needed to continue the run, written atomically. """
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'curr_iter': self.curr_iter,
            'rng_torch': torch.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        path = os.path.join(self.ckpt_dir, 'last.pt')
        torch.save(state, path + '.tmp')
        os.replace(path + '.tmp', path)

    def _maybe_resume(self):
        """ Loads `last.pt` if `trainer.resume` is set and it exists. Returns
        the epoch to continue with. """
        path = os.path.join(self.ckpt_dir, 'last.pt')
        if not (self.resume and os.path.exists(path)):
            if self.resume:
                print(f'trainer.resume is set but {path} does not exist; starting from scratch.')
            return 0
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.curr_iter = state['curr_iter']
        torch.set_rng_state(state['rng_torch'].cpu())
        if state['rng_cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in state['rng_cuda']])
        self.history = self._read_history(state['epoch'] + 1)
        return state['epoch'] + 1


class GICLTrainer(Trainer):
    """ `Trainer` for `ssl_neuron.giclmorph.GICLMorph`.

    Same schedule, optimizer, logging and checkpointing; the batch additionally
    carries one 2D projection per cell, and the IMID and CMID parts of the loss
    are reported separately (`imid`, `cmid` columns of `metrics.csv`).
    """
    def _forward(self, data):
        images = data[4].to(self.device, non_blocking=True)
        loss, parts, emb = self.model(*self._inputs(data), images, return_embedding=True)
        return loss, {'loss': loss.detach(), 'imid': parts['imid'], 'cmid': parts['cmid']}, emb
