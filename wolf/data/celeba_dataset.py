import glob
from torch.utils.data import Dataset
from torch import LongTensor
import torchvision.transforms as transforms
from PIL import Image



class CelebaDataset(Dataset):
    def __init__(self, data_path=None, label_path=None, image_size=32, split='train'):
        self.data_path = data_path
        self.label_path = label_path
        
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor()
            ])
        elif split == 'valid':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ])
        else:
            raise ValueError('The Celeb-A only accept split mode train and valid')

    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, idx):
        image_set = Image.open(self.data_path[idx])
        image_tensor = self.transform(image_set)
        image_label = LongTensor([0.] + self.label_path[idx])

        # return image_tensor, None

        return image_tensor, image_label