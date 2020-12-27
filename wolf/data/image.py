import os
import scipy.io
import numpy as np

import torch
from torchvision import datasets, transforms

import glob
from zipfile import ZipFile

import random
from .celeba_dataset import CelebaDataset, CelebaHairDataset
from .brats import BratsDataSet


def load_datasets(dataset, image_size, data_path):
    if dataset == 'omniglot':
        return load_omniglot()
    elif dataset == 'mnist':
        return load_mnist()
    elif dataset.startswith('lsun'):
        category = None if dataset == 'lsun' else dataset[5:]
        return load_lsun(data_path, category, image_size)
    elif dataset == 'cifar10':
        return load_cifar10(data_path)
    elif dataset == 'imagenet':
        return load_imagenet(data_path, image_size)
    elif dataset == 'celeba':
        return load_celeba(data_path, image_size)
    elif dataset == 'celeba-hair':
        return load_celeba_hair(data_path, image_size)
    elif dataset == 'brats':
        return load_brats(data_path, image_size)
    else:
        raise ValueError('unknown data set %s' % dataset)


def load_omniglot():
    def reshape_data(data):
        return data.T.reshape((-1, 1, 28, 28))

    omni_raw = scipy.io.loadmat('data/omniglot/chardata.mat')

    train_data = reshape_data(omni_raw['data']).astype(np.float32)
    train_label = omni_raw['target'].argmax(axis=0)
    test_data = reshape_data(omni_raw['testdata']).astype(np.float32)
    test_label = omni_raw['testtarget'].argmax(axis=0)

    train_data = torch.from_numpy(train_data).float()
    train_label = torch.from_numpy(train_label).long()
    test_data = torch.from_numpy(test_data).float()
    test_label = torch.from_numpy(test_label).long()

    return [(train_data[i], train_label[i]) for i in range(len(train_data))], \
           [(test_data[i], test_label[i]) for i in range(len(test_data))]


def load_mnist():
    train_data, train_label = torch.load('data/mnist/processed/training.pt')
    test_data, test_label = torch.load('data/mnist/processed/test.pt')

    train_data = train_data.float().div(256).unsqueeze(1)
    test_data = test_data.float().div(256).unsqueeze(1)

    return [(train_data[i], train_label[i]) for i in range(len(train_data))], \
           [(test_data[i], test_label[i]) for i in range(len(test_data))]


def load_lsun(data_path, category, image_size):
    if category is None:
        classes_train = 'train'
        classes_val = 'val'
    else:
        classes_train = [category + '_train']
        classes_val = [category + '_val']
    train_data = datasets.LSUN(data_path, classes=classes_train,
                               transform=transforms.Compose([
                                   transforms.CenterCrop(256),
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                               ]))

    val_data = datasets.LSUN(data_path, classes=classes_val,
                             transform=transforms.Compose([
                                 transforms.CenterCrop(256),
                                 transforms.Resize(image_size),
                                 transforms.ToTensor(),
                             ]))
    return train_data, val_data


def load_cifar10(data_path):
    imageSize = 32
    train_data = datasets.CIFAR10(data_path, train=True,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.Pad(4, padding_mode='reflect'),
                                      transforms.RandomCrop(imageSize),
                                      transforms.RandomHorizontalFlip(0.5),
                                      transforms.ToTensor()
                                  ]))
    test_data = datasets.CIFAR10(data_path, train=False,
                                 transform=transforms.Compose([
                                     transforms.ToTensor()
                                 ]))
    return train_data, test_data


def load_imagenet(data_path, image_size):
    data_path = os.path.join(data_path, 'imagenet{}x{}'.format(image_size, image_size))
    train_data = datasets.ImageFolder(os.path.join(data_path, 'train'),
                                      transform=transforms.Compose([
                                          transforms.ToTensor()
                                      ]))
    val_data = datasets.ImageFolder(os.path.join(data_path, 'val'),
                                    transform=transforms.Compose([
                                        transforms.ToTensor()
                                    ]))
    return train_data, val_data


def load_brats(data_path, image_size):
    train_data = BratsDataSet(data_path=data_path, image_size=image_size, split='train')
    valid_data = BratsDataSet(data_path, image_size, split='valid')

    return train_data, valid_data


def load_celeba_hair(data_path, image_size):
    image_path = os.path.join(data_path, 'img_hair_celeba/')
    attr_file = os.path.join(data_path, 'list_attr_celeba_hair.txt')
    if os.path.isdir(image_path) == 0:
        raise FileNotFoundError('The hair celeba data does not exist')
    
    data_paths = sorted(glob.glob(image_path + '*.jpg'))
    
    # get first 1000
    data_paths = data_paths[:1000]

    print('The number of extraction is {}'.format(len(data_paths)))

    label_list = open(attr_file).readlines()[1:]

    data_label = {}

    for label_str in label_list:
        img_name, label = label_str.split()
        img_file_name = os.path.join(image_path, img_name)
        label = int(label)
        data_label[img_file_name] = label
    
    indices = list(range(len(data_paths)))
    random.shuffle(indices)
    split_train = int(0.8 * len(data_paths))
    train_idx, valid_idx = indices[:split_train], indices[split_train:]

    train_data_paths = [data_paths[idx] for idx in train_idx]
    train_data_labels = [data_label[img_file] for img_file in train_data_paths]

    valid_data_paths = [data_paths[idx] for idx in valid_idx]
    valid_data_labels = [data_label[img_file] for img_file in valid_data_paths]

    # build up the dataset
    train_data = CelebaHairDataset(
        data_path=train_data_paths,
        label_path=train_data_labels,
        image_size=image_size,
        split='train',
    )

    val_data = CelebaHairDataset(
        data_path=valid_data_paths,
        label_path=valid_data_labels,
        image_size=image_size,
        split='valid',
    )

    return train_data, val_data
    

def load_celeba(data_path, image_size):
    image_file = os.path.join(data_path, 'img_align_celeba.zip')
    attr_file = os.path.join(data_path, 'list_attr_celeba.txt')
    direct_folder = os.path.join(data_path, 'img_align_celeba/')

    with ZipFile(image_file, 'r') as z:
        if os.path.isdir(direct_folder) == 0:
            print('Extracting Celeb-A files now...')
            z.extractall(data_path)
            print('Done!')
        else:
            print('Celeb-A has already been extracted.')
    
    data_paths = sorted(glob.glob(direct_folder+'*.jpg'))

    # here we only select first 5000
    data_paths = data_paths[:500]

    print('The number of extraction is {}'.format(len(data_paths)))

    label_list = open(attr_file).readlines()[2:]

    data_label = []

    for i in range(len(label_list)):
        data_label.append(label_list[i].split())
    
    # transform label (1, -1) to (1, 0)
    for m in range(len(data_label)):
        data_label[m] = [n.replace('-1', '0') for n in data_label[m]][1:]
        data_label[m] = [int(p) * 1.0 for p in data_label[m]]
    
    attributes = open(attr_file).readlines()[1].split()

    # shuffle the index
    indices = list(range(len(data_paths)))
    random.shuffle(indices)
    split_train = int(0.8 * len(data_paths))
    train_idx, valid_idx = indices[:split_train], indices[split_train:]

    train_data_paths = [data_paths[idx] for idx in train_idx]
    train_labels = [data_label[idx] for idx in train_idx]

    valid_data_paths = [data_paths[idx] for idx in valid_idx]
    valid_labels = [data_label[idx] for idx in valid_idx]

    # build up the dataset
    train_data = CelebaDataset(
        data_path=train_data_paths,
        label_path=train_labels,
        image_size=image_size,
        split='train',
    )

    val_data = CelebaDataset(
        data_path=valid_data_paths,
        label_path=valid_labels,
        image_size=image_size,
        split='valid',
    )

    return train_data, val_data


def get_batch(data, indices):
    imgs = []
    labels = []
    for index in indices:
        img, label = data[index]
        imgs.append(img)
        labels.append(label)

    try:
        return torch.stack(imgs, dim=0), torch.LongTensor(labels)
    except:
        return torch.stack(imgs, dim=0), torch.stack(labels, dim=0)


def iterate_minibatches(data, indices, batch_size, shuffle):
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, len(indices), batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield get_batch(data, excerpt)


def binarize_image(img):
    return torch.rand(img.size()).type_as(img).le(img).float()


def binarize_data(data):
    return [(binarize_image(img), label) for img, label in data]


def preprocess(img, n_bits, noise=None):
    n_bins = 2. ** n_bits
    # rescale to 255
    img = img.mul(255)
    if n_bits < 8:
        img = torch.floor(img.div(256. / n_bins))

    if noise is not None:
        # [batch, nsamples, channels, H, W]
        img = img.unsqueeze(1) + noise
    # normalize
    img = img.div(n_bins)
    img = (img - 0.5).div(0.5)
    # img in the range (-1, 1)
    return img


def postprocess(img, n_bits):
    n_bins = 2. ** n_bits
    # re-normalize
    img = img.mul(0.5) + 0.5
    img = img.mul(n_bins)
    # scale
    img = torch.floor(img) * (256. / n_bins)
    img = img.clamp(0, 255).div(255)
    return img
