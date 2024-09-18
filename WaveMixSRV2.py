import torch
import torch.nn as nn
import wavemix
from wavemix import DWTForward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

xf1 = DWTForward(J=1, mode='zero', wave='db1').to(device)   

class WaveMixSRV2Block(nn.Module):
    def __init__(
        self,
        *,
        mult = 2,
        ff_channel = 16,
        final_dim = 16,
        dropout = 0.5,
    ):
        super().__init__()
        
      
        self.feedforward = nn.Sequential(
                nn.Conv2d(final_dim, final_dim*mult,1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv2d(final_dim*mult, ff_channel, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(final_dim, final_dim,3, 1, 1),
                nn.BatchNorm2d(final_dim)
            
            )

        self.reduction = nn.Conv2d(final_dim, int(final_dim/4), 1)
        
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        x = self.reduction(x)
        
        Y1, Yh = xf1(x)
        
        x = torch.reshape(Yh[0], (b, int(c*3/4), int(h/2), int(w/2)))
        
        x = torch.cat((Y1,x), dim = 1)
        
        x = self.feedforward(x)
        
        return x


class SR_Block(nn.Module):
    def __init__(
        self,
        *,
        depth,
        mult = 1,
        final_dim = 16,
        dropout = 0.3,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Level1Waveblock(mult = mult, final_dim = final_dim, dropout = dropout))
        
        self.final = nn.Sequential(
            nn.Conv2d(final_dim,int(final_dim/2), 3, stride=1, padding=1),
            nn.Conv2d(int(final_dim/2), 1, 1)
        )


        self.path1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners = False),
            nn.Conv2d(1, int(final_dim/2), 3, 1, 1),
            nn.Conv2d(int(final_dim/2), final_dim, 3, 1, 1)
        )

        self.path2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners = False),
        )

    def forward(self, img):

        y = img[:, 0:1, :, :] 
        crcb = img[:, 1:3, :, :]

        y = self.path1(y)

        for attn in self.layers:
            y = attn(y) + y

        y = self.final(y)

        crcb = self.path2(crcb)
        
        return  torch.cat((y,crcb), dim=1)


class WaveMixSR_V2(nn.Module):
    def __init__(
        self,
        *,
        sr = 2,
        blocks = 2,
        mult = 1,
        final_dim = 16,
        dropout = 0.3,
    ):
        super().__init__()
        
        self.SR_blocks = nn.ModuleList([])
        for _ in range(int(sr/2)):
            self.SR_blocks.append(SR_Block(depth = blocks, mult = mult, final_dim = final_dim, dropout = dropout))

    def forward(self, x):

        for sr_block in self.SR_blocks:
            x = sr_block(x) 

        return  x

model = WaveMixSR_V2(
    sr          = 8,  #SR task, 2x, 4x, etc
    blocks      = 2,  #WaveMixSR-V2 blocks inside each 2x SR Block
    mult        = 1,  #Channel expansion factor inside MLP
    final_dim   = 144,
    dropout     = 0.3
).to(device)
