import os
import sys
import random
import numpy as np
from PIL import Image
from pydensecrf import densecrf
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

def dense_crf(img, output_probs):
    h = output_probs.shape[0]
    w = output_probs.shape[1]
    
    output_probs = np.expand_dims(output_probs, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)
    
    d = densecrf.DenseCRF2D(w, h, 2)
    U = -np.log(output_probs)
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(img)
    
    d.setUnaryEnergy(U)
    
    d.addPairwiseGaussian(sxy=20, compat=3)
    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)
    
    Q = d.inference(5)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))
    
    return Q

class ImageDataset(Dataset):
    
    def __init__(self, fdir, bdir, imsize, mode, aug_prob):
        self._fdir = fdir
        self._imsize = imsize
        self._mode = mode
        self._rot_degs = [0, 90, 180, 270]
        self._aug_prob = aug_prob
        self._impaths = list(map(lambda x: os.path.join(fdir, x), os.listdir(fdir)))
        self._bgpaths = list(map(lambda x: os.path.join(bdir, x), os.listdir(bdir)))
        self._dataset = list()
        print("image count in {} path: {}".format(self._mode, len(self._impaths)))
        index = 0
        while index < len(self._impaths):
            oimg = Image.open(self._impaths[index])
            assert str(oimg.mode) == 'RGBA'
            x, y = oimg.size
            aspect_ratio = y / x
            ch_r, ch_g, ch_b, ch_a = oimg.split()
            img = Image.merge('RGB', (ch_r, ch_g, ch_b))
            mask = ch_a
            ''' Add background '''
            if random.random() < 0.75:
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
            else:
                index += 1
            Transform = list()
            Transform.append(T.Resize((self._imsize, self._imsize)))
            Transform.append(T.ToTensor())
            Transform = T.Compose(Transform)
            img = Transform(img)
            mask = Transform(mask)
            Norm = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            img = Norm(img)
            self._dataset.append((img, mask))
            ratio = int((index+1)*50/len(self._impaths))
            sys.stdout.write("\r["+">"*ratio+" "*(50-ratio)+"] {}/{} {:.2f}%".format(index+1, len(self._impaths), (index+1)*100/len(self._impaths)))
            sys.stdout.flush()
        print()
        print("augmented image count in {} dataset: {}".format(self._mode, len(self._dataset)))
    
    def __getitem__(self, index):
        return self._dataset[index]
    
    def __len__(self):
        return len(self._dataset)
    
class TestImageDataset(Dataset):
    
    def __init__(self, fdir, imsize):
        self._fdir = fdir
        self._imsize = imsize
        self._impaths = list(map(lambda x: os.path.join(fdir, x), os.listdir(fdir)))
        Transform = list()
        Transform.append(T.Resize((self._imsize, self._imsize)))
        Transform.append(T.ToTensor())
        Transform.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        Transform = T.Compose(Transform)
        self._dataset = list()
        self._osize = list()
        for file in self._impaths:
            img = Image.open(file)
            assert str(img.mode) == 'RGB'
            self._osize.append(img.size)
            img = Transform(img)
            self._dataset.append(img)
        print("image count in test path: {}".format(len(self._impaths)))
    
    def __getitem__(self, index):
        return index, self._dataset[index]
    
    def __len__(self):
        return len(self._impaths)
    
    def save_img(self, index, predict, use_crf):
        predict = predict.squeeze().cpu().numpy()
        if use_crf:
            img = self._dataset[index].permute(1, 2, 0).numpy()
            predict = dense_crf(np.array(img).astype(np.uint8), predict)
        predict = np.array((predict > 0.5) * 255).astype(np.uint8)
        mask = Image.fromarray(predict, mode='L')
        mask = mask.resize(self._osize[index])
        fg = Image.new('RGB', self._osize[index], (0, 0, 0))
        bg = Image.new('RGB', self._osize[index], (255, 255, 255))
        bg.paste(fg, mask=mask)
        bg.save('./predicts/{:s}'.format(os.path.split(self._impaths[index])[-1]))
    
