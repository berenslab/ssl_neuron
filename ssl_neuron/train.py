import os
import torch
import torch.optim as optim
from ssl_neuron.utils import AverageMeter, compute_eig_lapl_torch_batch

class Trainer(object):
    def __init__(self, config, model, dataloaders):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.config = config
        self.ckpt_dir = config['trainer']['ckpt_dir']
        self.save_every = config['trainer']['save_ckpt_every']

        ### datasets
        self.train_loader = dataloaders[0]
        self.val_loader= dataloaders[1]

        ### trainings params
        self.max_iter = config['optimizer']['max_iter']
        self.init_lr = config['optimizer']['lr']
        self.exp_decay = config['optimizer']['exp_decay']
        self.lr_warmup = torch.linspace(0., self.init_lr,  steps=(self.max_iter // 50)+1)[1:]
        self.lr_decay = self.max_iter // 5
        
        # `name` and `weight_decay` are optional; the defaults are GraphDINO's
        # original plain Adam without weight decay.
        optimizers = {'adam': optim.Adam, 'adamw': optim.AdamW}
        optimizer_cls = optimizers[config['optimizer'].get('name', 'adam')]
        self.optimizer = optimizer_cls(list(self.model.parameters()), lr=0,
                                       weight_decay=config['optimizer'].get('weight_decay', 0.0))
        
        
    def set_lr(self): 
        if self.curr_iter < len(self.lr_warmup):
            lr = self.lr_warmup[self.curr_iter]
        else:
            lr = self.init_lr * self.exp_decay ** ((self.curr_iter - len(self.lr_warmup)) / self.lr_decay)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr
        

    def train(self):     
        self.curr_iter = 0
        epoch = 0
        while self.curr_iter < self.max_iter:
            # Run one epoch.
            self._train_epoch(epoch)

            if epoch % self.save_every == 0:
                # Save checkpoint.
                self._save_checkpoint(epoch)
            
            epoch += 1

        # Always keep the final weights, not only the last multiple of
        # `save_ckpt_every`.
        if (epoch - 1) % self.save_every != 0:
            self._save_checkpoint(epoch - 1)


    def _train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()
        for i, data in enumerate(self.train_loader, 0):
            f1, f2, a1, a2 = [x.float().to(self.device, non_blocking=True) for x in data]
            n = a1.shape[0]

            # compute positional encoding
            l1 = compute_eig_lapl_torch_batch(a1)
            l2 = compute_eig_lapl_torch_batch(a2)
            
            self.lr = self.set_lr()
            self.optimizer.zero_grad(set_to_none=True)
            
            loss = self.model(f1, f2, a1, a2, l1, l2)

            # optimize 
            loss.sum().backward()
            self.optimizer.step()
            
            # update teacher weights
            self.model.update_moving_average()
            
            losses.update(loss.detach(), n)
            self.curr_iter += 1

        print('Epoch {} | Loss {:.4f}'.format(epoch, losses.avg))


    def _save_checkpoint(self, epoch):
        filename = 'ckpt_{}.pt'.format(epoch)
        PATH = os.path.join(self.ckpt_dir, filename)
        torch.save(self.model.state_dict(), PATH)
        print('Save model after epoch {} as {}.'.format(epoch, filename))

class GICLTrainer(Trainer):
    """ `Trainer` for `ssl_neuron.giclmorph.GICLMorph`.

    Same schedule, optimizer and checkpointing; the batch additionally carries
    one 2D projection per cell, and the IMID and CMID parts of the loss are
    reported separately. Per-epoch losses are appended to `history.csv` in the
    checkpoint directory.
    """
    def _train_epoch(self, epoch):
        self.model.train()
        meters = {'loss': AverageMeter(), 'imid': AverageMeter(), 'cmid': AverageMeter()}
        for i, data in enumerate(self.train_loader, 0):
            f1, f2, a1, a2 = [x.float().to(self.device, non_blocking=True) for x in data[:4]]
            images = data[4].to(self.device, non_blocking=True)
            n = a1.shape[0]

            # compute positional encoding
            l1 = compute_eig_lapl_torch_batch(a1)
            l2 = compute_eig_lapl_torch_batch(a2)

            self.lr = self.set_lr()
            self.optimizer.zero_grad(set_to_none=True)

            loss, parts = self.model(f1, f2, a1, a2, l1, l2, images)

            loss.backward()
            self.optimizer.step()

            # update teacher weights
            self.model.update_moving_average()

            meters['loss'].update(loss.item(), n)
            meters['imid'].update(parts['imid'].item(), n)
            meters['cmid'].update(parts['cmid'].item(), n)
            self.curr_iter += 1

        print('Epoch {} | iter {} | lr {:.2e} | Loss {:.4f} | IMID {:.4f} | CMID {:.4f}'.format(
            epoch, self.curr_iter, float(self.lr),
            meters['loss'].avg, meters['imid'].avg, meters['cmid'].avg))

        history = os.path.join(self.ckpt_dir, 'history.csv')
        write_header = not os.path.exists(history) or epoch == 0
        with open(history, 'w' if epoch == 0 else 'a') as f:
            if write_header:
                f.write('epoch,iter,lr,loss,imid,cmid\n')
            f.write('{},{},{},{},{},{}\n'.format(epoch, self.curr_iter, float(self.lr),
                                              meters['loss'].avg, meters['imid'].avg,
                                              meters['cmid'].avg))
