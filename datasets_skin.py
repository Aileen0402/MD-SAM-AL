import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class SkinLesionDataset(torch.utils.data.Dataset):
    """Skin lesion dataset with paired images and masks"""
    def __init__(self, image_dir, mask_dir, image_list=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        if image_list is None:
            all_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            self.image_list = [f for f in all_files if '_superpixels' not in f]
        else:
            self.image_list = [f for f in image_list if '_superpixels' not in f]

        def get_mask_name(image_name):
            return image_name.rsplit(".", 1)[0] + "_segmentation.png"

        self.image_list = [f for f in self.image_list if os.path.exists(os.path.join(mask_dir, get_mask_name(f)))]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')
        mask_name = img_name.rsplit(".", 1)[0] + "_segmentation.png"
        mask = Image.open(os.path.join(self.mask_dir, mask_name)).convert('L')
        sample = {'image': image, 'label': mask, 'name': img_name}
        if self.transform:
            sample = self.transform(sample)
        return sample


class SemiSupervisedSkinDataset(SkinLesionDataset):
    """Dataset supporting labeled/unlabeled samples"""
    def __init__(self, image_dir, mask_dir, image_list=None, labeled_images=None, transform=None):
        super().__init__(image_dir, mask_dir, image_list, transform)
        self.labeled_images = labeled_images or {}

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')
        if img_name in self.labeled_images:
            mask_name = img_name.rsplit(".", 1)[0] + "_segmentation.png"
            mask = Image.open(os.path.join(self.mask_dir, mask_name)).convert('L')
        else:
            mask = Image.new('L', image.size, 0)
        sample = {'image': image, 'label': mask, 'name': img_name, 'is_labeled': img_name in self.labeled_images}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ValSkinDataset(SkinLesionDataset):
    """Validation dataset (same as training dataset)"""
    def __getitem__(self, idx):
        return super().__getitem__(idx)


class SkinTrainGenerator(object):
    """Transform for training images and masks"""
    def __init__(self, output_size=[512, 512]):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = Image.fromarray(np.array(image)).resize(self.output_size, Image.BILINEAR)
        label = Image.fromarray(np.array(label)).resize(self.output_size, Image.NEAREST)
        image = transforms.ToTensor()(image)
        label = torch.from_numpy(np.array(label)).float()
        label = (label > 127).float()
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])(image)
        return {'image': image, 'label': label}


class SkinValGenerator(SkinTrainGenerator):
    """Transform for validation images and masks"""
    def __call__(self, sample):
        return super().__call__(sample)
