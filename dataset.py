import torchvision.transforms.functional as F
import numpy as np
import random
import os
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ToTensor(object):

    def __call__(self, data):
        image, label = data['image'], data['label']
        # 图像转换为 [0,1] 范围
        image = F.to_tensor(image)
        # 掩码转换为 [0,1] 范围，然后二值化
        label = F.to_tensor(label)
        label = (label > 0.5).float()  # 确保掩码是二值的
        return {'image': image, 'label': label}


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']

        return {'image': F.resize(image, self.size), 'label': F.resize(label, self.size, interpolation=InterpolationMode.BICUBIC)}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.hflip(image), 'label': F.hflip(label)}

        return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.vflip(image), 'label': F.vflip(label)}

        return {'image': image, 'label': label}


class RandomRotation(object):
    """随机旋转（90/180/270度），掩码同步旋转"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']
        if random.random() < self.p:
            angle = random.choice([90, 180, 270])
            image = F.rotate(image, angle)
            label = F.rotate(label, angle)
        return {'image': image, 'label': label}


class ColorJitter(object):
    """
    颜色抖动（只作用于图像，不影响掩码）
    叶片病害图像颜色变化大（光照、品种差异），颜色增强对泛化很重要
    """
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.5):
        self.p = p
        self.jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, data):
        image, label = data['image'], data['label']
        if random.random() < self.p:
            image = self.jitter(image)
        return {'image': image, 'label': label}


class RandomGrayscale(object):
    """随机灰度化（模拟不同光照条件），只作用于图像"""
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']
        if random.random() < self.p:
            image = F.to_grayscale(image, num_output_channels=3)
        return {'image': image, 'label': label}


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # 只对图像进行标准化，掩码保持 [0,1] 范围
        image = F.normalize(image, self.mean, self.std)
        # 确保掩码是二值的
        label = (label > 0.5).float()
        return {'image': image, 'label': label}
    

class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, mode):
        # Get all image files
        image_files = [f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        mask_files = [f for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.jpg')]
        
        # Create matching pairs
        self.images = []
        self.gts = []
        
        for img_file in sorted(image_files):
            # Try to find corresponding mask with same name (any extension)
            img_name = os.path.splitext(img_file)[0]
            mask_found = False
            
            # Try .png first, then .jpg
            for ext in ['.png', '.jpg']:
                mask_name = img_name + ext
                if mask_name in mask_files:
                    self.images.append(os.path.join(image_root, img_file))
                    self.gts.append(os.path.join(gt_root, mask_name))
                    mask_found = True
                    break
            
            if not mask_found:
                print(f"Warning: No mask found for image {img_file}")
        
        print(f"Dataset loaded: {len(self.images)} matching image-mask pairs")
        if len(image_files) != len(self.images):
            print(f"Warning: {len(image_files) - len(self.images)} images have no corresponding masks")
        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                RandomRotation(p=0.5),           # 新增：随机旋转 90/180/270
                ColorJitter(                      # 新增：颜色抖动（叶片病害颜色变化大）
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.05,
                    p=0.5
                ),
                RandomGrayscale(p=0.1),           # 新增：随机灰度化（模拟不同光照）
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])
        data = {'image': image, 'label': label}
        data = self.transform(data)
        return data['image'], data['label']

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        

class TestDataset:
    def __init__(self, image_root, gt_root, size):
        # Get all image files
        image_files = [f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        mask_files = [f for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.jpg')]
        
        # Create matching pairs
        self.images = []
        self.gts = []
        
        for img_file in sorted(image_files):
            # Try to find corresponding mask with same name (any extension)
            img_name = os.path.splitext(img_file)[0]
            mask_found = False
            
            # Try .png first, then .jpg
            for ext in ['.png', '.jpg']:
                mask_name = img_name + ext
                if mask_name in mask_files:
                    self.images.append(os.path.join(image_root, img_file))
                    self.gts.append(os.path.join(gt_root, mask_name))
                    mask_found = True
                    break
            
            if not mask_found:
                print(f"Warning: No mask found for image {img_file}")
        
        print(f"TestDataset loaded: {len(self.images)} matching image-mask pairs")
        if len(image_files) != len(self.images):
            print(f"Warning: {len(image_files) - len(self.images)} images have no corresponding masks")
        
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        if self.index >= self.size:
            raise IndexError(f"Dataset index {self.index} out of range. Dataset size: {self.size}")
        
        if self.index >= len(self.images):
            raise IndexError(f"Image index {self.index} out of range. Images count: {len(self.images)}")
        
        if self.index >= len(self.gts):
            raise IndexError(f"Mask index {self.index} out of range. Masks count: {len(self.gts)}")
            
        try:
            image = self.rgb_loader(self.images[self.index])
            image = self.transform(image).unsqueeze(0)

            gt = self.binary_loader(self.gts[self.index])
            gt = np.array(gt)

            name = os.path.basename(self.images[self.index])

            self.index += 1
            return image, gt, name
        except Exception as e:
            print(f"Error loading data at index {self.index}: {e}")
            print(f"Image path: {self.images[self.index] if self.index < len(self.images) else 'N/A'}")
            print(f"Mask path: {self.gts[self.index] if self.index < len(self.gts) else 'N/A'}")
            raise
    
    def reset(self):
        """Reset the dataset index to 0"""
        self.index = 0

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')