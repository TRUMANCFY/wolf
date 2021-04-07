from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from random import sample


# why we need to prepare (sort the data here)
def prepare_data(cfg, data):
    imgs, captions, captions_lens = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = torch.sort(captions_lens, 0, True)
    
    sorted_cap_lens = sorted_cap_lens.squeeze()

    sorted_cap_indices = sorted_cap_indices.squeeze()

    real_imgs = imgs[sorted_cap_indices].cuda()

    captions = captions[sorted_cap_indices].squeeze()

    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    return real_imgs, captions, sorted_cap_lens


def get_imgs(cfg, img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size

    if transform is not None:
        img = transform(img)
    
    if normalize is not None:
        img = normalize(img)

    return img


class TextDataset(data.Dataset):
    def __init__(self, data_dir, cfg, imsize, split='train',
                 base_size=64,
                 transform=None, target_transform=None):
        self.cfg = cfg
        # convert the image to the target size
        self.transform = transforms.Compose([
            transforms.Scale(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        ## Not sure whether we need this preprocessing normalization
        self.norm = None
        self.target_transform = target_transform
        self.embeddings_num = self.cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imsize = imsize

        self.data = []
        self.data_dir = data_dir
        self.bbox = None

        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                print("filepath", filepath)
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names

        return filenames, captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding='latin1')
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        x = np.zeros((self.cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= self.cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:self.cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = self.cfg.TEXT.WORDS_NUM
        return x, [x_len]

    def __getitem__(self, index):

        key = self.filenames[index]
        cls_id = self.class_id[index]

        data_dir = '%s/CUB_200_2011' % self.data_dir

        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(self.cfg, img_name, self.imsize,
                        self.bbox, self.transform, normalize=self.norm)
        # randomly select a sentence
        sent_ix = random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(new_sent_ix)

        imgs_tensor = torch.Tensor(imgs)
        caps_tensor = torch.Tensor(caps)
        cap_len_tensor = torch.Tensor(cap_len)

        # here we currently set the label to be the length
        return imgs_tensor, cap_len_tensor

        # return imgs_tensor, caps_tensor, cap_len_tensor

    def __len__(self):
        return len(self.filenames)

    def ixs2words(self, token_list):
        res_sent = ''
        for token in token_list:
            res_sent += self.ixtoword(token)
        return res_sent

    def selectRandomCaption(self, num=10):
        inds = random.sample(range(len(self.filenames)), num)
        caps = []
        cap_lens = []
        cap_txts = []
        imgs = []

        data_dir = '%s/CUB_200_2011' % self.data_dir

        for ind in inds:
            key = self.filenames[index]

            sent_ix = random.randint(0, self.embeddings_num)
            new_sent_ix = ind * self.embeddings_num + new_sent_ix
            caps, cap_len = get_caption(new_sent_ix)
            # convert caps to cap
            token_list = caps.flatten().tolist()
            cap_txts.append(res_sent)

            # load image
            img_name = '%s/images/%s.jpg' % (data_dir, key)
            img = get_imgs(self.cfg, img_name, self.imsize,
                        self.bbox, self.transform, normalize=self.norm)

            imgs.append(img)

        imgs = torch.Tensor(imgs)
        caps = torch.Tensor(caps)
        cap_lens = torch.Tensor(cap_lens)

        return imgs, caps, cap_lens, cap_txts