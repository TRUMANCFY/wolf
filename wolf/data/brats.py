from torch.utils.data import Dataset
from torch import LongTensor
import torch
import os
import numpy as np
import torchvision.transforms as transforms


class BratsDataSet(Dataset):
    def __init__(self, data_path=None, label_path=None, image_size=32, split='train'):
        self.data_path = os.path.join(data_path, split, 'data.pkl')
        self.label_path = os.path.join(data_path, split, 'label.pkl')
        self.seg_path = os.path.join(data_path, split, 'seg.pkl')

        # read the data
        self.data = np.load(self.data_path)
        self.label = np.load(self.label_path)
        self.seg = np.load(self.seg_path)
        self.seg = self.seg.astype('float32')
        print('data: ', self.data.dtype)
        print('seg: ', self.seg.dtype)

        assert self.data.shape[0] == self.label.shape[0] == self.seg.shape[0], 'The length of data, label and segmentation should be the same'

        # convert numpy to tensor
        self.data = torch.from_numpy(self.data)
        self.label = torch.from_numpy(self.label)
        self.seg = torch.from_numpy(self.seg)

        self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ])

        print('BraTS generation Split: {}, #Data: {}'.format(split, self.data.shape[0]))

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        image_tensor = self.data[idx, :, :]
        image_tensor = torch.stack([image_tensor, image_tensor])
        image_label = self.label[idx]
        seg_tensor = self.seg[idx, :, :]

        image_tensor = self.transform(image_tensor)
        seg_tensor = self.transform(seg_tensor)

        return image_tensor, image_label

