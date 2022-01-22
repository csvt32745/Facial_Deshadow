import torch
from torch import nn
from src.defineHourglass_512_gray_skip import BasicBlock

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
        return self
    
    def loadModel(self, path):
        print(f"Network Loading: {path}")
        self.net.load_state_dict(torch.load(path))
        return self
    
    def __call__(self, *args, **kwargs):
        return self.net(*args, **kwargs)
    
    def get_features(self, *args, **kwargs):
        ''' return (n_stacks, batch, ch_out, h, w), (n_stacks, batch, ch_bottleneck, h, w) '''
        return self.net.get_features(*args, **kwargs)
    
    def get_bottlenecks(self, *args, **kwargs):
        ''' return (n_stacks, batch, ch_bottleneck, h, w) '''
        return self.net.get_bottlenecks(*args, **kwargs)
    
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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        def conv3x3(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                        stride=stride, padding=1, bias=False)
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1, True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(self.skip(residual))
        out += residual
        out = self.relu(out)
        return out

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
            # sequence += [
            #     nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            #     norm_layer(ndf * nf_mult),
            #     nn.LeakyReLU(0.2, True)
            # ]
            sequence.append(ResidualBlock(ndf * nf_mult_prev, ndf * nf_mult, stride=2, downsample=nn.MaxPool2d(2, 2)))

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


class DomainClassifier(nn.Module):
    def __init__(self, ch_in=256, w=16):
        super().__init__()
        ch1 = ch_in*2
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch1, 3, 1, 1), # 16
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(ch1), 
            nn.MaxPool2d(2, stride=2), # 8

            nn.Conv2d(ch1, ch1, 3, 1, 1), # 8
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(ch1), 
            nn.MaxPool2d(2, stride=2), # 4

            nn.Conv2d(ch1, ch1, 3, 1, 1), # 4
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(ch1), 
            nn.MaxPool2d(2, stride=2), # 2
            
            nn.Flatten(), # 2*2*512
        )
        ch_f1 = ch1*(w//8)*(w//8) # 512*4
        ch_f2 = ch_f1//4 # 256
        ch_f3 = ch_f2//4 # 32
        ch_f4 = ch_f3//4 # 32
        self.fc = nn.Sequential(
            nn.Linear(ch_f1, ch_f2),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm1d(ch_f2),

            nn.Linear(ch_f2, ch_f3),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm1d(ch_f3),

            nn.Linear(ch_f3, ch_f4),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm1d(ch_f4),

            nn.Linear(ch_f4, 1),
        )
        
    def forward(self, x):
        return self.fc(self.conv(x))