import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import make_grid

from einops import rearrange
from tqdm import tqdm

from src.Score import DictLosser
from src.network import Net
from src.utils import denormalize_img_rgb, TorchLab_RGB255
from src.logger import WBLogger

class BasicGANTrainer():
    def __init__(self, 
        logger: WBLogger, model_name,
        G: Net, D: Net,
        train_loader, valid_loader, unsup_loader,
        pretrain, epoch, now_time, log_interval=100,
        save_path="./models", test_loader=None, is_rgb=False, n_stacks=1, add_shadow_weight=0.):
        
        self.logger = logger
        self.model_name = model_name
        self.now_time = now_time
        self.epoch = epoch
        self.pretrain_epoch = pretrain
        self.best_loss = 1e9

        self.D = D
        self.G = G

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.unsup_loader = unsup_loader
        self.test_loader = test_loader

        self.losser = DictLosser()
        self.image_buffer = {}

        self.iteration = 0
        self.logging_interval = log_interval
        self.logging_validimg_num = 3

        self.net_save_path = os.path.join(
            save_path, self.model_name, self.now_time)

        self.logger.LogNet(self.G.net, 'G')
        self.logger.LogNet(self.D.net, 'D')

        # from high to low, open_skip == N means openning the N-th inner layer
        self.open_skip_epoch = [6, 10, 13, 15][::-1]
        # self.open_skip_epoch = [10, 15, 20, 25][::-1]
        self.open_skip = len(self.open_skip_epoch)

        # get the required channels (Luminance from Lab, or RGB)
        self.is_rgb = is_rgb
        self.GetChs = (lambda x: x.cuda()) if is_rgb else (lambda x: x[:, [0]].cuda())

        self.n_stacks = max(n_stacks, 1)
        self.add_shadow_weight = add_shadow_weight

    def apply_weight(self, mask):
        return mask*self.add_shadow_weight + 1.

    def merge_image(self, x, f, r, num=1):
        img = rearrange(
            torch.stack((x[:num], f[:num], r[:num])),
            'n b c h w -> (b h) (n w) c'
        )
        img = (img.detach().numpy()*255).astype(np.uint8) if self.is_rgb else TorchLab_RGB255(img)
        return img

    def SaveModel(self, save_path, file_name):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        rgb_str = '_RGB' if self.is_rgb else ''
        self.D.saveModel(os.path.join(save_path, f"{file_name+rgb_str}_D.pt"))
        self.G.saveModel(os.path.join(save_path, f"{file_name+rgb_str}_G.pt"))
        self.logger.info("Model save in "+ save_path)

    def CalOpenSkip(self, epoch):
        if epoch >= self.open_skip_epoch[self.open_skip-1] and self.open_skip > 0:
            self.open_skip -= 1

    def Train(self):
        self.logger.info(f"Pretrain for {self.pretrain_epoch:02d} epochs")
        for epoch in range(self.pretrain_epoch):
            self.Pretrain(epoch)

        self.logger.info("Training")
        for epoch in range(self.epoch):
            self.CalOpenSkip(epoch)
            self.logger.info("Skip count: %d" % self.open_skip)
            self.EpochTrain(epoch)
            self.EpochValidation(epoch)
            self.EpochTest(epoch)
            if (epoch+1) % 10 == 0 or (epoch+1) == self.epoch:
                self.SaveModel(self.net_save_path, f"{epoch}")

    def Pretrain(self, epoch):
        self.D.train()
        self.G.train()
        self.losser.clear()
        for i, x in enumerate(tqdm(self.unsup_loader, position=0, leave=True, dynamic_ncols=True)):
            # x_, real_, sh = x
            # x = self.GetChs(x_)
            x = self.GetChs(x)
            
            # Train D
            fake = self.G.get_features(x, skip_count=999)
            # lossG_recon = crit(fake, x)
            # loss = self.G.crit.recon(torch.cat(fake), torch.tile(x, (self.n_stacks, 1, 1, 1))).mean()
            loss = self.G.crit.compute_all_woadv(torch.cat(fake), torch.tile(x, (self.n_stacks, 1, 1, 1)), ret_dict=False)[0]
            self.G.step(loss)
            
            # Record
            self.losser.add({'Recon': loss.item()})

            if (i+1) % self.logging_interval == 0:
                self.logger.LogTrainingDB(
                    epoch, self.pretrain_epoch, i, len(self.unsup_loader),
                    self.iteration, self.losser.mean(), is_update=False)
                self.losser.clear()
    

    def EpochTrain(self, epoch):
        self.D.train()
        self.G.train()
        self.losser.clear()

        for i, (x, u) in enumerate(zip(
            tqdm(self.train_loader, position=0, leave=True, dynamic_ncols=True),
            self.unsup_loader
            )):
            x_, real_, shadow_mask, sh = x
            
            real = self.GetChs(real_)
            x = self.GetChs(x_)
            u = self.GetChs(u)
            # real = real_[:, [0]].cuda()
            # x = x_[:, [0]].cuda()
            # u = u[:, [0]].cuda()
            shadow_mask = shadow_mask.cuda()
            sh = sh.cuda()
            

            # Train D
            fake = self.G.get_features(x, skip_count=self.open_skip)
            fake_result = fake[-1]
            fake = torch.cat(fake)

            prob_real = self.D(real)
            prob_fake = self.D(fake.detach())
            # prob_real = self.D(u)

            lossD_real = self.D.crit(prob_real, torch.ones_like(prob_real, device='cuda'))
            lossD_fake = self.D.crit(prob_fake, torch.zeros_like(prob_fake, device='cuda'))
            loss = lossD_fake + lossD_real
            self.D.step(loss)

            # Train G
            prob = self.D(fake)
            loss, loss_dict = self.G.crit(
                fake, torch.tile(real, (self.n_stacks, 1, 1, 1)), prob,
                weight=torch.tile(self.apply_weight(shadow_mask), (self.n_stacks, 1, 1, 1)))
            # lossG_adv = self.D.crit(prob, label_real)
            # loss = lossG_recon + lossG_adv
            self.G.step(loss)
            
            # Record
            loss_dict.update({
                "D_real": lossD_real.item(),
                "D_fake": lossD_fake.item(),
                # "G_adv": lossG_adv.item()
            })
            self.losser.add(loss_dict)

            self.iteration += 1
            
            if (i+1) % self.logging_interval == 0:
                # substitute the lum channel -> shadowed and deshadowed img
                if self.is_rgb:
                    orig = x_
                    fake_ = fake_result.detach().cpu()
                else:
                    orig = real_.detach().cpu().clone() 
                    fake_ = orig.clone()
                    orig[:, 0] = x_[:, 0]
                    fake_[:, 0] = fake_result[:, 0]

                self.image_buffer['Sample'] = self.merge_image(orig, fake_, real_, num=1)
                
                self.logger.LogTrainingDB(
                    epoch, self.epoch, i, len(self.train_loader),
                    self.iteration, self.losser.mean(), self.image_buffer)
                self.losser.clear()
                self.image_buffer.clear()
            
        self.D.step_scheduler()
        self.G.step_scheduler()

    @torch.no_grad()
    def EpochValidation(self, epoch):
        self.D.eval()
        self.G.eval()
        self.losser.clear()

        log_flg = True
        for i, (x, u) in enumerate(zip(
            tqdm(self.valid_loader, position=0, leave=True, dynamic_ncols=True),
            self.unsup_loader
            )):
            x_, real_, shadow_mask, sh = x
            
            real = self.GetChs(real_)
            x = self.GetChs(x_)
            u = self.GetChs(u)
            # real = real_[:, [0]].cuda()
            # x = x_[:, [0]].cuda()
            # u = u[:, [0]].cuda()
            shadow_mask = shadow_mask.cuda()
            sh = sh.cuda()
            
            fake = self.G.get_features(x, skip_count=self.open_skip)
            fake_result = fake[-1]
            fake = torch.cat(fake)

            prob_real = self.D(real)
            # prob_real = self.D(u)
            prob_fake = self.D(fake)
            lossD_real = self.D.crit(prob_real, torch.ones_like(prob_real, device='cuda'))
            lossD_fake = self.D.crit(prob_fake, torch.zeros_like(prob_fake, device='cuda'))

            _, loss_dict = self.G.crit(
                fake, torch.tile(real, (self.n_stacks, 1, 1, 1)), prob_fake,
                weight=torch.tile(self.apply_weight(shadow_mask), (self.n_stacks, 1, 1, 1)))
            
            # Record
            loss_dict.update({
                "D_real": lossD_real.item(),
                "D_fake": lossD_fake.item(),
            })
            self.losser.add(loss_dict)

            if log_flg:
                if self.is_rgb:
                    orig = x_
                    fake_ = fake_result.detach().cpu()
                else:
                    # substitute the lum channel -> shadowed and deshadowed img
                    orig = real_.detach().cpu().clone() 
                    fake_ = orig.clone()
                    orig[:, 0] = x_[:, 0]
                    fake_[:, 0] = fake_result[:, 0]
            
                self.image_buffer['Sample'] = self.merge_image(orig, fake_, real_, num=self.logging_validimg_num)
                log_flg = False
            
        self.logger.LogValidationDB(epoch, self.epoch, self.iteration, self.losser.mean(), self.image_buffer)
        self.image_buffer.clear()

    @torch.no_grad()
    def EpochTest(self, epoch):
        if self.test_loader is None:
            return
        self.D.eval()
        self.G.eval()
        
        fakes = []
        inputs = []
        for i, x in enumerate(tqdm(self.test_loader, position=0, leave=True, dynamic_ncols=True)):
            fake = x.clone()
            res = self.G(self.GetChs(x), skip_count=self.open_skip).cpu()
            if self.is_rgb:
                fake = res
            else:
                fake[:, [0]] = res
            
            inputs.append(x.clone())
            fakes.append(fake.clone())
        inputs = torch.cat(inputs)
        fakes = torch.cat(fakes)
        
        output = make_grid(rearrange(
            torch.stack([inputs, fakes]),
            "col b c h w -> b c h (col w)"
            ),
            nrow=4
        )
        self.logger.LogImageDB(self.iteration, 
            (output.permute(1, 2, 0).numpy()*255).astype(np.uint8) if self.is_rgb
            else TorchLab_RGB255(output.permute(1, 2, 0))
        )
            
            
        