import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from einops import rearrange
from tqdm import tqdm

from src.Score import DictLosser
from src.network import Net
from src.utils import denormalize_img_rgb, TorchLab_RGB255
from src.logger import WBLogger

class BasicGANTrainer():
    def __init__(self, 
        config, logger: WBLogger, model_name,
        G: Net, D: Net,
        train_loader, valid_loader,
        pretrain, epoch, now_time, log_interval=100,
        save_path="./models"):
        
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

        self.losser = DictLosser()
        self.image_buffer = {}

        self.iteration = 0
        self.logging_interval = log_interval
        self.logging_validimg_num = 3

        self.net_save_path = os.path.join(
            save_path, self.model_name, self.now_time+self.model_name)

        self.logger.LogNet(self.G.net, 'G')
        self.logger.LogNet(self.D.net, 'D')

        self.open_skip_epoch = [10, 15, 20, 25][::-1]
        self.open_skip = len(self.open_skip_epoch)
        # from high to low, open_skip == N means openning the N-th inner layer

    def SaveModel(self, save_path, file_name):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        self.D.saveModel(os.path.join(save_path, f"{file_name}_D.pt"))
        self.G.saveModel(os.path.join(save_path, f"{file_name}_G.pt"))
        self.logger.info("Model save in "+ save_path)

    def CalOpenSkip(self, epoch):
        if epoch >= self.open_skip_epoch[self.open_skip-1] and self.open_skip > 0:
            self.open_skip -= 1

    def Train(self):
        self.logger.info("Pretrain for epochs")
        for epoch in range(self.pretrain_epoch):
            self.Pretrain(epoch)

        self.logger.info("Training")
        for epoch in range(self.epoch):
            self.CalOpenSkip(epoch)
            self.EpochTrain(epoch)
            self.EpochValidation(epoch)
            if (epoch+1) % 5 == 0 or (epoch+1) == self.epoch:
                self.SaveModel(self.net_save_path, f"{epoch}")

    def Pretrain(self, epoch):
        self.D.train()
        self.G.train()
        self.losser.clear()
        for i, x in enumerate(tqdm(self.train_loader, position=0, leave=True, dynamic_ncols=True)):
            x_, real_, sh = x
            real = real_[:, [0]].cuda()
            x = x_[:, [0]].cuda()
            b = x.size(0)
            
            # Train D
            fake = self.G(x, skip_count=0)
            # lossG_recon = crit(fake, x)
            loss = self.G.crit.recon(fake, x)
            self.G.step(loss)
            
            # Record
            self.losser.add({'Recon': loss.item()})

            if (i+1) % self.logging_interval == 0:
                self.logger.LogTrainingDB(
                    epoch, self.pretrain_epoch, i, len(self.train_loader),
                    self.iteration, self.losser.mean(), is_update=False)
                self.losser.clear()
    
    @staticmethod
    def merge_image(x, f, r, num=1):
        img = rearrange(
            torch.stack((x[:num], f[:num], r[:num])),
            'n b c h w -> (b h) (n w) c'
        )
        img = TorchLab_RGB255(img)
        return img
            
    def EpochTrain(self, epoch):
        self.D.train()
        self.G.train()
        self.losser.clear()

        for i, x in enumerate(tqdm(self.train_loader, position=0, leave=True, dynamic_ncols=True)):
            x_, real_, sh = x
            real = real_[:, [0]].cuda()
            x = x_[:, [0]].cuda()
            sh = sh.cuda()

            # Train D
            fake = self.G(x, skip_count=self.open_skip)
            prob_real = self.D(real)
            prob_fake = self.D(fake.detach())
            label_fake = torch.zeros_like(prob_fake, device='cuda')
            label_real = torch.ones_like(prob_fake, device='cuda')

            lossD_real = self.D.crit(prob_real, label_real)
            lossD_fake = self.D.crit(prob_fake, label_fake)
            loss = lossD_fake + lossD_real
            self.D.step(loss)

            # Train G
            prob = self.D(fake)
            loss, loss_dict = self.G.crit(fake, real, prob)
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
                fake_ = x_.detach().cpu().clone()
                fake_[:, 0] = fake[:, 0]
                self.image_buffer['Sample'] = self.merge_image(x_, fake_, real_, num=1)
                
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
        for i, x in enumerate(tqdm(self.valid_loader, position=0, leave=True, dynamic_ncols=True)):
            x_, real_, sh = x
            real = real_[:, [0]].cuda()
            x = x_[:, [0]].cuda()
            sh = sh.cuda()
            
            fake = self.G(x, skip_count=self.open_skip)
            prob_real = self.D(real)
            prob_fake = self.D(fake)
            lossD_real = self.D.crit(prob_real, torch.ones_like(prob_fake, device='cuda'))
            lossD_fake = self.D.crit(prob_fake, torch.zeros_like(prob_fake, device='cuda'))
            _, loss_dict = self.G.crit(fake, real, prob_fake)
            
            # Record
            loss_dict.update({
                "D_real": lossD_real.item(),
                "D_fake": lossD_fake.item(),
            })
            self.losser.add(loss_dict)

            if log_flg:
                fake_ = x_.detach().cpu().clone()
                fake_[:, 0] = fake[:, 0]
                self.image_buffer['Sample'] = self.merge_image(x_, fake_, real_, num=self.logging_validimg_num)
                log_flg = False
            
        self.logger.LogValidationDB(epoch, self.epoch, self.iteration, self.losser.mean(), self.image_buffer)
        self.image_buffer.clear()