from torch import Tensor, nn
import torch

class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)

class Discriminator(nn.Module):
    def __init__(self, ndf, in_channel=3):
        super().__init__()
        # self.fcs_in = nn.Linear(6, 257)
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, ndf, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.PReLU(ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf*2, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*2, affine=True),
            nn.PReLU(2*ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf*2, ndf*4, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*4, affine=True),
            nn.PReLU(4*ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf*4, ndf*8, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*8, affine=True),
            nn.PReLU(8*ndf),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf*8, ndf*4)),
            nn.Dropout(0.3),
            nn.PReLU(4*ndf),
            nn.utils.spectral_norm(nn.Linear(ndf*4, 1)),
            LearnableSigmoid(1)
        )

    def forward(self, x, y, ht):
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        # ht_embed = self.fcs_in(ht).unsqueeze(1)
        xy = torch.cat([x, y], dim=1)
        xy_ht = torch.cat([xy, ht], dim=1)
        return self.layers(xy_ht)


if __name__ == '__main__':
    inputs1 = torch.randn(1, 100, 257)
    inputs2 = torch.randn(1, 100, 257)
    inputs3 = torch.randn(1, 1, 100, 257)


    Model = Discriminator(ndf=16)

    predict_metric = Model(inputs1, inputs2, inputs3)

    print(predict_metric.shape)
