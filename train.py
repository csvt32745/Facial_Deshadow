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
import wandb

from config import config_parser_train
from src.logger import WBLogger
from src.dataset import DPRShadowDataset, DPRShadowDataset_ColorJitter, UnsupervisedDataset
from src.train_code import BasicGANTrainer
from src.network import *
from src.defineHourglass_512_gray_skip import HourglassNet, HourglassNetExquotient, RecursiveStackedHourglassNet, SimpleStackedHourglassNet
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
    log_name = os.path.join(config.save_log_path, config.exp_name, now_time + ".log")
    logger = WBLogger(
        config.exp_name, log_path=log_name, level=logging.DEBUG, config=config
    )  
    config = wandb.config

    if config['kernelratio_low'] > config['kernelratio_high']:
        tmp = config['kernelratio_low']
        config['kernelratio_low'] = config['kernelratio_high']
        config['kernelratio_high'] = tmp
        wandb.config.update(config)

    LR = config['learning_rate']
    EPOCH = config['epoch']
    PRETRAIN_EPOCH = config['pretrain_epoch']
    BATCH_SIZE = config['batch_size']
    IS_RGB = config['is_rgb'] or config['is_colorjitter']
    N_STACKS = config['n_stacks']
    IS_RECUR_STACK = config['is_recursive_stack']
    IS_OUT_GAIN_BIAS = config['is_out_gainbias']

    opt_func = lambda net: optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr = LR)
    # sch_func = lambda opt: optim.lr_scheduler.MultiStepLR(opt, milestones=[5], gamma = 0.2)
    sch_func = lambda opt: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCH, eta_min=1e-6, verbose=True)
    
    ch_in = 3 if IS_RGB else 1
    ch_bottleneck = 32
    G_NET = HourglassNetExquotient if IS_OUT_GAIN_BIAS else HourglassNet
    G = Net(
        G_NET(ch_in=ch_in, baseFilter=ch_bottleneck) if N_STACKS <= 1 \
        # HourglassNet(ch_in=ch_in, baseFilter=ch_bottleneck) if N_STACKS <= 1 \
            else RecursiveStackedHourglassNet(
                n_stacks=N_STACKS, ch_in=ch_in, baseFilter=ch_bottleneck, net_class=G_NET
            ) if IS_RECUR_STACK \
            else SimpleStackedHourglassNet(
                n_stacks=N_STACKS, ch_in=ch_in, baseFilter=ch_bottleneck, net_class=G_NET
            ),
        opt_func, GeneratorLoss(is_rgb=IS_RGB), scheduler_func=sch_func
    )
    D = Net(
        NLayerDiscriminator(ch_in, ndf=64),
        # ResNet(models.resnet18(pretrained=True)),
        # TEST nn.BCEWithLogitsLoss() -> MSE
        opt_func, nn.MSELoss(), scheduler_func=sch_func
    )
    try:
        if config['model_path']:
            G = G.loadModel(config['model_path'] + "G.pt") # RGB
            D = D.loadModel(config['model_path'] + "D.pt") # RGB
        else:
            if N_STACKS < 1:
                if IS_RGB:
                    G = G.loadModel('models/2021_1214_05:31:35/9_G.pt') # RGB
                else:
                    G = G.loadModel('models/2021_1202_12:57:15/4_G.pt') # lab
            elif N_STACKS == 2:
                G = G.loadModel('models/stacked/2021_1221_01:56:28/9_RGB_G.pt')
            else:
                G = G.loadModel('models/stacked/2021_1221_01:58:22/9_RGB_G.pt')
    except:
        logger.warning("Load model error, using default model.")
    
    G = G.cuda()
    D = D.cuda()

    def get_dataloader(batch_size, shuffle, n_workers, dataset):
        dataloader = DataLoader(dataset, batch_size, shuffle, num_workers=n_workers, pin_memory=True)
        return dataloader

    n_workers = 16
    data_split = json.load(open(os.path.join(config['dataset_path'], config['data_split_path']))) if config['dataset_path'] else None
    if config['smaller_dataset'] < 1.:
        for k in data_split:
            data_split[k] = data_split[k][:int(config['smaller_dataset']*len(data_split[k]))]

    DPRDATASET = DPRShadowDataset_ColorJitter if config['is_colorjitter'] else DPRShadowDataset
    train_loader = get_dataloader(BATCH_SIZE, True, n_workers,
        DPRDATASET(config['dataset_path'], data_split['train'] if data_split else None,
            k_size=(config['kernelratio_low'], config['kernelratio_high']),
            intensity=(config['intensity_low'], config['intensity_high']),
            is_rgb=IS_RGB,
        )
    )
    valid_loader = get_dataloader(BATCH_SIZE, False, n_workers, 
        DPRDATASET(config['dataset_path'], data_split['valid'] if data_split else None,
            k_size=(config['kernelratio_low'], config['kernelratio_high']),
            intensity=(config['intensity_low'], config['intensity_high']),
            is_rgb=IS_RGB,
        )
    )
    unsup_loader = get_dataloader(BATCH_SIZE, True, n_workers, UnsupervisedDataset('ffhq', is_rgb=IS_RGB))
    test_loader = get_dataloader(BATCH_SIZE, False, n_workers, UnsupervisedDataset('ffhq_test', is_rgb=IS_RGB))

    logger.LogTrainValid(len(data_split['train']), len(data_split['valid']))

    trainer = BasicGANTrainer(
        logger, config['exp_name'],
        G, D, train_loader, valid_loader, unsup_loader,
        PRETRAIN_EPOCH, EPOCH,
        now_time, log_interval=len(train_loader)//2, 
        save_path=config['save_model_path'], test_loader=test_loader, 
        is_rgb=IS_RGB, n_stacks=N_STACKS, add_shadow_weight=config['add_shadow_weight']
    )
    trainer.Train()
    
if __name__ == '__main__':
    set_seed(52728)
    main(config_parser_train())

    