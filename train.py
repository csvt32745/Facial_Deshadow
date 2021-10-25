import sys
import os
import logging
import random
import json
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from torch.utils.data.dataloader import DataLoader

from config import config_parser_train
from src.logger import WBLogger
from src.dataset import DPRShadowDataset
from src.train_code import BasicGANTrainer
from src.network import *
from src.defineHourglass_512_gray_skip import HourglassNet
from src.loss_func import GeneratorLoss

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def main(config):
    now_time = datetime.now().strftime("%Y_%m%d_%I:%M:%S")
    LR = config.learning_rate
    EPOCH = config.epoch
    BATCH_SIZE = config.batch_size

    log_name = os.path.join(config.save_log_path, config.exp_name, now_time + ".log")
    logger = WBLogger(
        config.exp_name, log_path=log_name, level=logging.DEBUG, config=config
    )

    opt_func = lambda net: optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr = LR)
    # sch_func = lambda opt: optim.lr_scheduler.MultiStepLR(opt, milestones=[5], gamma = 0.2)
    sch_func = lambda opt: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.epoch, eta_min=1e-6, verbose=True)

    G = Net(
        HourglassNet(ch_in=1, ch_out=1, baseFilter=32),
        opt_func, GeneratorLoss(), scheduler_func=sch_func
    )
    D = Net(
        NLayerDiscriminator(1, ndf=64),
        # ResNet(models.resnet18(pretrained=True)),
        opt_func, nn.BCEWithLogitsLoss(), scheduler_func=sch_func
    )
    G = G.cuda()
    D = D.cuda()

    def get_dataloader(batch_size, shuffle, n_workers, root, img_list=None):
        dataset = DPRShadowDataset(root, img_list)
        dataloader = DataLoader(dataset, batch_size, shuffle, num_workers=n_workers, pin_memory=True)
        return dataloader

    n_workers = 16
    data_split = json.load(open(os.path.join(config.dataset_path, config.data_split_path))) if config.dataset_path else None
    train_loader = get_dataloader(BATCH_SIZE, True, n_workers, config.dataset_path, data_split['train'] if data_split else None)
    valid_loader = get_dataloader(BATCH_SIZE, False, n_workers, config.dataset_path, data_split['valid'] if data_split else None)

    
    trainer = BasicGANTrainer(
        config, logger, config.exp_name,
        G, D, train_loader, valid_loader,
        0, EPOCH,
        now_time, log_interval=1, 
        save_path=config.save_model_path
    )
    trainer.Train()
    
if __name__ == '__main__':
    set_seed(52728)
    main(config_parser_train())

    