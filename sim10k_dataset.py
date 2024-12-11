import os
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
import cv2
import csv

class Sim10kDataset(Dataset):
    def __init__(self, img_dir, segm_dir, transform=None, threshold=False, cache=True):
        """
        Initialize the Sim10k dataset.

        Args:
            img_dir (str): Path to the image directory.
            segm_dir (str): Path to the segmentation mask directory.
            transform (callable, optional): Transform to be applied to images and masks.
            threshold (bool, optional): Whether to apply threshold adjustment to bounding boxes.
            cache (bool, optional): Whether to cache data to speed up reading.
        """
        self.img_dir = img_dir
        self.segm_dir = segm_dir
        self.transform = transform
        self.threshold = threshold
        self.cache = cache
        self.masks = []

        # Read valid .seg file paths from CSV file
        csv_filename = '/data/cxli/yuzhi/Ambiguous_SAM/filtered_valid_masks.csv'
        with open(csv_filename, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                mask_path = row[0]  # Assume the first column in the CSV file is the path
                mask_name = os.path.basename(mask_path)  # Get the file name (excluding the path)
                self.masks.append(mask_name)

        # Update self.imgs dictionary, mapping .seg file names read from the CSV file to corresponding .jpg image file names
        self.imgs = {os.path.splitext(m)[0]: os.path.splitext(m)[0] + '.jpg' for m in self.masks}

        # Cache dictionary
        self.cache_data = {}

    def square_crop_around_foreground(self, image, box, img_height, img_width):
        x_min, y_min, x_max, y_max = box
        crop_size = 128
        center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2

        new_x_min = max(0, center_x - crop_size // 2)
        new_y_min = max(0, center_y - crop_size // 2)
        new_x_max = new_x_min + crop_size
        new_y_max = new_y_min + crop_size

        if new_x_max > img_width:
            new_x_min = img_width - crop_size
            new_x_max = img_width
        if new_y_max > img_height:
            new_y_min = img_height - crop_size
            new_y_max = img_height

        cropped_image = image[new_y_min:new_y_max, new_x_min:new_x_max]
        return cropped_image

    def get_bbox(self, label):
        gt2D = np.asarray(label, dtype="uint8")
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        return np.array([x_min, y_min, x_max, y_max])

    def adjust_bbox(self, bbox, scale_min=0.6, scale_max=1.5, shift_max=10):
        W_gt, H_gt = 128, 128
        x_min, y_min, x_max, y_max = bbox
        width, length = x_max - x_min, y_max - y_min
        scale_factor = random.uniform(scale_min, scale_max)
        width, length = width * scale_factor, length * scale_factor
        center_x, center_y = x_min + width / 2, y_min + length / 2
        shift_x, shift_y = random.randint(-shift_max, shift_max), random.randint(-shift_max, shift_max)
        center_x += shift_x
        center_y += shift_y
        new_x_min, new_y_min = max(0, center_x - width / 2), max(0, center_y - length / 2)
        new_x_max, new_y_max = min(W_gt, center_x + width / 2), min(H_gt, center_y + length / 2)
        return np.array([new_x_min, new_y_min, new_x_max, new_y_max])

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        mask_name = self.masks[idx]
        img_name = self.imgs[os.path.splitext(mask_name)[0]]
        img_path = os.path.join(self.img_dir, img_name)
        segm_path = os.path.join(self.segm_dir, mask_name)

        if self.cache and img_path in self.cache_data:
            image, mask_A, mask_B, mask_AB, box_final, box_shift = self.cache_data[img_path]
        else:
            image = Image.open(img_path).convert('RGB')
            segmentation = Image.open(segm_path).convert('RGB')
            image = np.array(image)
            segmentation = np.array(segmentation)

            seg_array = segmentation
            unique_colors = np.unique(seg_array.reshape(-1, 3), axis=0)
            non_black_colors = unique_colors[np.any(unique_colors != [0, 0, 0], axis=1)]

            if len(non_black_colors) < 2:
                raise ValueError("There should be at least two non-black colors in the segmentation mask.")

            mask_A = (seg_array == non_black_colors[0]).all(axis=-1).astype(np.int64)
            mask_B = (seg_array == non_black_colors[1]).all(axis=-1).astype(np.int64)
            mask_AB = np.maximum(mask_A, mask_B)

            ys, xs = np.where(mask_AB > 0)
            ymin, ymax, xmin, xmax = ys.min(), ys.max(), xs.min(), xs.max()
            extended_box = [xmin, ymin, xmax, ymax]

            image = self.square_crop_around_foreground(image, extended_box, 1052, 1914)
            mask_A = self.square_crop_around_foreground(mask_A, extended_box, 1052, 1914)
            mask_B = self.square_crop_around_foreground(mask_B, extended_box, 1052, 1914)
            mask_AB = self.square_crop_around_foreground(mask_AB, extended_box, 1052, 1914)

            image = cv2.resize(image, (128, 128))
            mask_A = cv2.resize(mask_A, (128, 128), interpolation=cv2.INTER_NEAREST)
            mask_B = cv2.resize(mask_B, (128, 128), interpolation=cv2.INTER_NEAREST)
            mask_AB = cv2.resize(mask_AB, (128, 128), interpolation=cv2.INTER_NEAREST)

            box_final = self.get_bbox(mask_AB)
            box_shift = self.adjust_bbox(box_final) if not self.threshold else np.array([box_final[0] - 5, box_final[1] - 5, box_final[2] + 5, box_final[3] + 5])

            if self.cache:
                self.cache_data[img_path] = (image, mask_A, mask_B, mask_AB, box_final, box_shift)

        masks = [mask_A, mask_B, mask_AB]
        mask = masks[random.randint(0, 2)]
        mask = torch.from_numpy(mask).long()
        masks = np.stack(masks, axis=0)

        W_gt, H_gt = 128, 128
        box_1024 = box_final / np.array([W_gt, H_gt, W_gt, H_gt]) * 1024
        box_1024 = np.array([box_1024])
        box_shift_1024 = box_shift / np.array([W_gt, H_gt, W_gt, H_gt]) * 1024
        box_shift_1024 = np.array([box_shift_1024])

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std

        image_normalized = torch.from_numpy(image).permute(2, 0, 1).float()

        if self.transform is not None:
            image = self.transform(image)

        sample = {
            'image': image_normalized,
            'label': mask,
            'label_four': torch.from_numpy(masks).long(),
            'box_1024': torch.from_numpy(box_1024).float(),
            'box_shift': torch.from_numpy(box_shift_1024).float(),
            'box_ori': torch.from_numpy(box_shift).float()
        }

        return sample