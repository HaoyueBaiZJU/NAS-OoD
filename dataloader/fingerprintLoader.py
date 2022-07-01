import os
from torchvision import datasets
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch.utils.data as data
from PIL import Image
import numpy as np

import torch.nn.functional as F

class FingerDataset(data.Dataset):
    
    def __init__(self, setname, args):
        
        if setname != 'test':
            domain = ['domain_4', 'domain_5', 'domain_6']
            domain.remove(args.fptargetdomain)   
            fulldata = []
            fullconcept = []
            i = 0
            for domain_name in domain:
                file_path = os.path.join(args.root_path, domain_name, setname)
                img_paths = self.parse_path(file_path)
                concept = [i] * len(img_paths)
                fulldata.extend(img_paths)
                fullconcept.extend(concept)
                i += 1
            self.data = fulldata
            self.concept = fullconcept
        else:
            fulldata = []
            fullconcept = []
            domain = args.fptargetdomain
            full = ['train', 'val', 'test']
            i = 0
            for typename in full:
                file_path = os.path.join(args.root_path, domain, typename)
                img_paths = self.parse_path(file_path)
                concept = [i] * len(img_paths)
                fulldata.extend(img_paths)
                fullconcept.extend(concept)
            self.data = fulldata
            self.concept = fullconcept
            
        self.num_class = 2
        self.num_concept = 2

        
    def __getitem__(self, i):
        img_pth = self.data[i]     

        img_origi = np.load(img_pth)
        img_arr = np.zeros((256, 256, 2))
        img_arr[:250, :250, :] = img_origi
        
        img_arr = torch.from_numpy(img_arr).float()
        img_arr /=255.
        img_arr = np.transpose(img_arr, (2,0,1))
        
        img_arr = F.interpolate(img_arr, size=(256, 256), mode='bicubic', align_corners=False)
        
        label = int(img_pth.split("_")[-1].split(".npy")[0])
        label_arr = torch.tensor(label)
        concept = self.concept[i]
        concept_arr = torch.tensor(concept)
        return img_arr, label_arr, concept_arr

    def parse_path(self,data_root):

        files = os.listdir(data_root)
        img_pths = [os.path.join(data_root,f) for f in files]
        return img_pths
    
    def __len__(self):
        
        return len(self.data)





