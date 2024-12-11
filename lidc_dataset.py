import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import random
import pickle
import os
import random
import numpy as np
import torch
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from einops import repeat
import numpy as np
import random

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label,label_four,box_1024,box_shift= sample['image'], sample['label'],sample['label_four'],sample['box_1024'],sample['box_shift']
        box_ori=sample['box_ori']
        label_four=np.stack(label_four,axis=0)
        label_four=label_four.astype(np.int64)
        label=label.squeeze()
        image_oc=image.copy()
        x, y = image.shape[-2:]
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        label_h, label_w = label.shape
        image = torch.from_numpy(image.astype(np.float32))
        image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        label = torch.from_numpy(label.astype(np.float32))
        box_1024 = torch.from_numpy(box_1024.astype(np.float32))
        box_shift = torch.from_numpy(box_shift.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'image_oc':image_oc,'label_four':label_four,'box_1024':box_1024,
                  'box_shift':box_shift,'box_ori':box_ori}
        return sample

class LIDC_IDRI(Dataset):
    images = []
    labels = []
    series_uid = []

    def __init__(self, dataset_location, transform=None,prior=False,threshold=False):
        self.transform = transform
        self.prior = prior
        self.threshold=threshold
        max_bytes = 2**31 - 1
        data = {}
        for file in os.listdir(dataset_location):
            filename = os.fsdecode(file)
            if '.pickle' in filename:
                print("Loading file", filename)
                file_path = dataset_location + filename
                bytes_in = bytearray(0)
                input_size = os.path.getsize(file_path)
                with open(file_path, 'rb') as f_in:
                    for _ in range(0, input_size, max_bytes):
                        bytes_in += f_in.read(max_bytes)
                new_data = pickle.loads(bytes_in)
                data.update(new_data)
        
        for key, value in data.items():
            self.images.append(value['image'].astype(float))
            self.labels.append(value['masks'])
            self.series_uid.append(value['series_uid'])

        assert (len(self.images) == len(self.labels) == len(self.series_uid))

        for img in self.images:
            assert np.max(img) <= 1 and np.min(img) >= 0
        for label in self.labels:
            assert np.max(label) <= 1 and np.min(label) >= 0

        del new_data
        del data


    def __len__(self):
        return len(self.images)

    def get_bbox(self,label):
        gt2D = np.asarray(label, dtype="uint8")
        H_gt, W_gt = gt2D.shape  # Height and width of the ground truth
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        bboxes = np.array([x_min-2, y_min-2, x_max+2, y_max+2])

        return bboxes



    def adjust_bbox(self,bbox, scale_min=0.6, scale_max=1.5, shift_max=10):
        """
        Adjust the bounding box of the labeled object by applying random scaling and translation factors.

        Args:
            label (np.array): The label image array where objects are marked.
            scale_min (float): Minimum scale factor.
            scale_max (float): Maximum scale factor.
            shift_max (int): Maximum shift in pixels in both x and y directions.

        Returns:
            np.array: Adjusted bounding box coordinates as [x_min, y_min, x_max, y_max].
        """
        W_gt=128
        H_gt=128
        # Extract coordinates from the input bbox
        x_min, y_min, x_max, y_max = bbox

        # Calculate the current width and length
        width = x_max - x_min
        length = y_max - y_min

        # Generate random scaling factor
        scale_factor = random.uniform(scale_min, scale_max)

        # Apply random scaling
        width = width * scale_factor
        length = length * scale_factor

        # Recalculate the center of the bbox
        center_x = x_min + width / 2
        center_y = y_min + length / 2

        # Generate random shift in X and Y direction
        shift_x = random.randint(-shift_max, shift_max)
        shift_y = random.randint(-shift_max, shift_max)

        # Apply random translation
        center_x += shift_x
        center_y += shift_y

        # Calculate new bbox coordinates
        new_x_min = center_x - width / 2
        new_y_min = center_y - length / 2
        new_x_max = center_x + width / 2
        new_y_max = center_y + length / 2

        # Ensure the bbox does not go out of image boundaries
        new_x_min = max(0, new_x_min)
        new_y_min = max(0, new_y_min)
        new_x_max = min(W_gt, new_x_max)
        new_y_max = min(H_gt, new_y_max)

        shifted_bbox = np.array([new_x_min, new_y_min, new_x_max, new_y_max])
        return shifted_bbox




    
    def __getitem__(self, index):#label随意 但是box一定得是最大的再去偏移
        all_boxes=[]
        image = np.expand_dims(self.images[index], axis=0)
        #Randomly select one of the four labels for this image

        label_four=self.labels[index]
        for label_iter in self.labels[index]:
            if label_iter.sum()!=0:
                box_iter=self.get_bbox(label_iter)
                all_boxes.append(box_iter)
        widest_longest_bbox = max(all_boxes, key=lambda bbox: bbox[2] - bbox[0] + bbox[3] - bbox[1])
        # print(widest_longest_bbox)
        label = self.labels[index][random.randint(0,3)].astype(float)
        if label.sum()==0:
            bboxes=widest_longest_bbox
        else:
            bboxes=self.get_bbox(label)
        if self.threshold:
            bboxes_shift=np.array([bboxes[0]-5, bboxes[1]-5, bboxes[2]+5, bboxes[3]+5])
        else:
            bboxes_shift = self.adjust_bbox(widest_longest_bbox)
        W_gt=128
        H_gt=128
        box_1024 = bboxes / np.array([W_gt, H_gt, W_gt, H_gt]) * 1024
        box_1024 = np.array([box_1024])
        bboxes_shift_1024=bboxes_shift/ np.array([W_gt, H_gt, W_gt, H_gt]) * 1024
        bboxes_shift_1024 = np.array([bboxes_shift_1024])
        # Convert image and label to torch tensors
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        #Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        label = label.type(torch.LongTensor)
        label=label.unsqueeze(0)
        image = np.array(image)
        label=np.array(label)
        # print(label.shape)
        sample={'image':image, 'label':label,'label_four':label_four,'box_1024':box_1024,'box_shift':bboxes_shift_1024,
                'box_ori':bboxes_shift}
        if self.transform:
            sample = self.transform(sample)

        return sample
