import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

class ImageDataset(Dataset):
    
    def __init__(self, fdir, bdir, imsize, mode, aug_prob):
        self._fdir = fdir
        self._imsize = imsize
        self._mode = mode
        self._rot_degs = [0, 90, 180, 270]
        self._aug_prob = aug_prob
        self._impaths = list(map(lambda x: os.path.join(fdir, x), os.listdir(fdir)))
        self._bgpaths = list(map(lambda x: os.path.join(bdir, x), os.listdir(bdir)))
        print("image count in {} path: {}".format(self._mode, len(self._impaths)))
    
    def __getitem__(self, index):
        oimg = Image.open(self._impaths[index])
        assert str(oimg.mode) == 'RGBA'
        x, y = oimg.size
        aspect_ratio = y / x
        ch_r, ch_g, ch_b, ch_a = oimg.split()
        img = Image.merge('RGB', (ch_r, ch_g, ch_b))
        mask = ch_a
        ''' Add background '''
        if random.random() < 0.5:
            bg = Image.new('RGB', img.size, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            bg.paste(img, mask=mask)
        else:
            bg = Image.open(self._bgpaths[random.randint(0, len(self._bgpaths)-1)])
            bg = bg.resize(img.size)
            bg.paste(img, mask=mask)
        img = bg
        ''' Do transformation '''
        if self._mode == 'train' and random.random() < self._aug_prob:
            Transform = list()
            resize_range = random.randint(300, 320)
            Transform.append(T.Resize((int(resize_range * aspect_ratio), resize_range)))
            rot_deg = self._rot_degs[random.randint(0, 3)]
            if rot_deg == 90 or rot_deg == 270:
                aspect_ratio = 1 / aspect_ratio
            Transform.append(T.RandomRotation((rot_deg, rot_deg)))
            rot_range = random.randint(-10, 10)
            Transform.append(T.RandomRotation((rot_range, rot_range)))
            crop_range = random.randint(250, 270)
            Transform.append(T.CenterCrop((int(crop_range * aspect_ratio), crop_range)))
            Transform = T.Compose(Transform)
            img = Transform(img)
            mask = Transform(mask)
            Transform = T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.02)
            img = Transform(img)
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() < 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
        Transform = list()
        Transform.append(T.Resize((self._imsize, self._imsize)))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        img = Transform(img)
        mask = Transform(mask)
        Norm = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        img = Norm(img)
        return img, mask
    
    def __len__(self):
        return len(self._impaths)
    
class TestImageDataset(Dataset):
    
    def __init__(self, fdir, imsize):
        self._fdir = fdir
        self._imsize = imsize
        self._impaths = list(map(lambda x: os.path.join(fdir, x), os.listdir(fdir)))
        print("image count in test path: {}".format(len(self._impaths)))
    
    def __getitem__(self, index):
        img = Image.open(self._impaths[index])
        assert str(img.mode) == 'RGB'
        Transform = list()
        Transform.append(T.Resize((self._imsize, self._imsize)))
        Transform.append(T.ToTensor())
        Transform.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        Transform = T.Compose(Transform)
        img = Transform(img)
        return index, img
    
    def __len__(self):
        return len(self._impaths)
    
    def save_img(self, index, predict):
        oimg = Image.open(self._impaths[index])
        x, y = oimg.size
        fg = Image.new('RGB', oimg.size, (0, 0, 0))
        bg = Image.new('RGB', oimg.size, (255, 255, 255))
        mask = Image.fromarray(np.uint8(predict * 255), mode='L')
        mask = mask.resize(oimg.size)
        bg.paste(fg, mask=mask)
        bg.save('./predicts/{:s}'.format(os.path.split(self._impaths[index])[-1]))
    
