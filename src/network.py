import torch
from torch import nn

class Net():
    def __init__(self,
        net: nn.Module,
        optimizer_func,
        criterion: nn.Module,
        scheduler_func = None,
    ):
        self.net = net
        self.optim = optimizer_func(net)
        self.scheduler = scheduler_func(self.optim) if scheduler_func else None
        self.crit = criterion

    def saveModel(self, path):
        torch.save(self.net.state_dict(), path)
    
    def loadModel(self, path):
        self.net.load_state_dict(torch.load(path))
    
    def __call__(self, *args, **kwargs):
        return self.net(*args, **kwargs)
    
    def train(self):
        self.net.train()
    
    def eval(self):
        self.net.eval()
    
    def step(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
    
    def step_scheduler(self):
        if self.scheduler:
            self.scheduler.step()

    def cuda(self):
        self.net = self.net.cuda()
        self.crit = self.crit.cuda()
        return self


class ResNet(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.resnet = resnet
        
        self.fc = nn.Sequential(
            nn.Linear(1000, 256),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm1d(256),

            # nn.Linear(256, 128),
            # nn.LeakyReLU(0.1, True),
            # nn.BatchNorm1d(128),

            # nn.Linear(128, 64),
            # nn.LeakyReLU(0.1, True),
            # nn.BatchNorm1d(64),

            nn.Linear(256, 1),
        )
        
    def forward(self, x):
        return self.fc(self.resnet(x))

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""
    ''' https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py '''
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)