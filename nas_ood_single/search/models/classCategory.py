import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class cNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            from ista_nas_single.search.models.convnet import ConvNet
            if args.dataset == 'finger':
                self.encoder = ConvNet(x_dim=2, hid_dim=64, z_dim=64)
                hdim = 576
            elif args.dataset == 'pacs':
                self.encoder = ConvNet()
                hdim = 256
            else:
                self.encoder = ConvNet()
                hdim = 64
        else:
            raise ValueError('')
            
        hdim_embs = 64
        self.fc = nn.Linear(hdim, hdim_embs)
        self.fc0 = nn.Linear(hdim_embs, args.num_class)

 
    def forward(self, x):
        
        embs = self.encoder(x)
        instance_embs = self.fc(embs)
        logits = self.fc0(instance_embs)

        return logits, instance_embs





