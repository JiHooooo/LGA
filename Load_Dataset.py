# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
from scipy import ndimage
from bert_embedding import BertEmbedding

#Convert image to torch.Tensor and divide by 255 if image or mask are uint8 type.
# from albumentations.pytorch import ToTensor
import cv2
import numpy as np

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text = sample['image'], sample['label'], sample['text']
        image, label = image.astype(np.uint8), label.astype(np.uint8)
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        text = torch.Tensor(text)
        sample = {'image': image, 'label': label, 'text': text}
        return sample


class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text = sample['image'], sample['label'], sample['text']
        image, label = image.astype(np.uint8), label.astype(np.uint8)  # OSIC
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        text = torch.Tensor(text)
        sample = {'image': image, 'label': label, 'text': text}
        return sample


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class LV2D(Dataset):
    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.output_path = os.path.join(dataset_path)
        self.mask_list = os.listdir(self.output_path)
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        self.bert_embedding = BertEmbedding()

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.output_path))

    def __getitem__(self, idx):

        mask_filename = self.mask_list[idx]  # Co
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1
        mask = correct_dims(mask)
        text = self.rowtext[mask_filename]
        text = text.split('\n')
        text_token = self.bert_embedding(text)
        text = np.array(text_token[0][1])
        if text.shape[0] > 14:
            text = text[:14, :]
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        sample = {'label': mask, 'text': text}

        return sample, mask_filename


class ImageToImage2D(Dataset):

    def __init__(self, dataset_path: str, text_path: str, task_name: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224, mean_text_flag = True) -> None:
        self.dataset_path = dataset_path
        self.text_path = text_path
        self.image_size = image_size
        self.task_name = task_name
        if self.task_name == 'Covid19':
            self.input_path = os.path.join(dataset_path, 'img')
            self.output_path = os.path.join(dataset_path, 'labelcol')
        elif self.task_name == 'MosMed':
            self.input_path = os.path.join(dataset_path, 'image')
            self.output_path = os.path.join(dataset_path, 'mask')

        self.images_list = os.listdir(self.input_path)
        self.mask_list = os.listdir(self.output_path)
        self.one_hot_mask = one_hot_mask
        
        self.mean_text_flag = mean_text_flag

        
        # self.bert_embedding = BertEmbedding()

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):

        # image_filename = self.images_list[idx]  # MoNuSeg
        # mask_filename = image_filename[: -3] + "png"  # MoNuSeg
        if self.task_name == 'Covid19':
            mask_filename = self.mask_list[idx]  # Covid19
            base_name = os.path.splitext(mask_filename)[0]
            image_filename = mask_filename.replace('mask_', '')  # Covid19
            image_path = os.path.join(self.input_path, image_filename)
            mask_path = os.path.join(self.output_path, mask_filename)
            text_path = '%s/%s.npy'%(self.text_path, base_name)
        elif self.task_name == 'MosMed':
            mask_filename = self.mask_list[idx]
            image_filename = mask_filename
            base_name = os.path.splitext(mask_filename)[0]
            image_path = os.path.join(self.input_path, mask_filename)
            mask_path = os.path.join(self.output_path, mask_filename)
            text_path = '%s/%s.npy'%(self.text_path, base_name)
        
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.image_size, self.image_size))

        # read mask image
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask[mask <= 0] = 0
        mask[mask > 1] = 1
        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
        # text = self.rowtext[mask_filename]
        # text = text.split('\n')
        # text_token = self.bert_embedding(text)
        # text = np.array(text_token[0][1])
        
        # # if text.shape[0] > 10:
        # #    text = text[:10, :]

        # # use average
        # text = np.mean(text, axis=0, keepdims=True)
        # text = np.load('%s/%s.npy'%(self.text_path, mask_filename))
        if self.mean_text_flag:
            text = np.mean(np.load(text_path), axis=0, keepdims=True)
        else:
            text = np.load(text_path)
        
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        if self.joint_transform:
            transformed = self.joint_transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # 
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        sample = {'image': self.image2tensor(image), 'label': self.mask2tensor(mask), 'text': torch.from_numpy(text.astype(np.float32))}

        return sample, image_filename

    def image2tensor(self, image):
        image = image / 255.0
        if len(image.shape) == 2:
            image = image[None]
        elif len(image.shape) == 3:
            image = image.transpose(2, 0, 1)
        else:
            raise ValueError('The size of input image should have two dimention or three dimention')
        return torch.from_numpy(image.astype(np.float32))

    def mask2tensor(self, mask):
        # multi_mask = np.zeros((self.label_num, mask.shape[0], mask.shape[1]), dtype=np.float32)
        # for index_mask in range(self.label_num):
        #     multi_mask[index_mask][mask == index_mask+1] = 1
        return (torch.from_numpy(mask)).long()