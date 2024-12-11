import os
import logging
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from utils import (
    generalized_energy_distance_iou, hm_iou_cal, dice_max_cal2
)
from lidc_data import LIDC_IDRI, RandomGenerator
from asam import ASAM

# Configure logging
logging.basicConfig(filename='evaluation_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Evaluate the model with specified epochs and weights.')
parser.add_argument('--epochs', nargs='+', type=int, default=[10, 20, 30, 50], help='Epochs to load weights from.')
parser.add_argument('--combined_weights_path', type=str, default='checkpoint/final_weights.pth', help='Path to the combined weights file.')
parser.add_argument('--gpuid', type=int, default=5, help='ID of the GPU to use.')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size for data loading.')
parser.add_argument('--total_samples', type=int, default=16, help='Total number of samples to generate.')
args = parser.parse_args()

# Set device
device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() else 'cpu')

# Load combined weights
combined_weights = torch.load(args.combined_weights_path, map_location=device)

# Initialize networks
def initialize_networks(epochs):
    networks = []
    for epoch in epochs:
        epoch_key = f'epoch_{epoch}'
        if epoch_key in combined_weights:
            net = ASAM().to(device)
            net.load_state_dict(combined_weights[epoch_key]['model_state_dict'])
            networks.append((net, combined_weights[epoch_key]['mask_weights'].to(device)))
        else:
            print(f"Warning: Weights for epoch {epoch} not found.")
    return networks

# Initialize scores
ged_score = dice_max2_score = hm_iou_score = 0
networks = initialize_networks(args.epochs)

# Log the weights being used
logging.info(f"Using combined weights from {args.combined_weights_path} for epochs: {args.epochs}")

# Prepare dataset
db = LIDC_IDRI(dataset_location='LIDC/data/', transform=transforms.Compose([
    RandomGenerator(output_size=[128, 128])
]))
dataset_size = len(db)
indices = list(range(dataset_size))
train_split = int(np.floor(0.6 * dataset_size))
validation_split = int(np.floor(0.8 * dataset_size))
test_indices = indices[validation_split:]

test_dataset = Subset(db, test_indices)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

print(f"Total dataset size: {dataset_size}")
print(f"Test set size: {len(test_indices)}")

# Hyperparameter: number of samples per network
samples_per_net = args.total_samples // len(networks)

# Evaluate the model
for i_batch, sampled_batch in enumerate(test_loader):
    print(f'Processing batch {i_batch}')
    logging.info(f'Processing batch {i_batch}')
    image_batch, label_batch = sampled_batch['image'].to(device), sampled_batch['label'].to(device)
    label_four_batch = sampled_batch['label_four']
    image_batch_oc = sampled_batch['image_oc'].to(device)
    box1024_batch = sampled_batch['box_1024'].to(device)
    boxshift_batch = sampled_batch['box_shift'].to(device)
    pred_list = [[] for _ in range(image_batch.shape[0])]

    for net, weights in networks:
        for _ in range(samples_per_net):  # Generate multiple samples per network
            outputs = net.forward(image_batch, image_batch_oc, box1024_batch, boxshift_batch, label_batch, device, train=False)
            logits_high = outputs['masks'].to(device) * weights.unsqueeze(-1)
            logits_high_res = logits_high.sum(1).unsqueeze(1)

            for j in range(image_batch.shape[0]):
                pred_list[j].append(logits_high_res[j])

    for index in range(len(pred_list)):
        pred_eval = torch.cat(pred_list[index], 0)
        pred_eval = (pred_eval > 0).cpu().detach().int()

        iou_score_iter, ged_score_iter = generalized_energy_distance_iou(pred_eval, label_four_batch[index])
        score = hm_iou_cal(pred_eval, label_four_batch[index])
        hm_iou_score += score
        dice_max2_score += dice_max_cal2(pred_eval, label_four_batch[index])
        ged_score += ged_score_iter

# Calculate average scores
ged = ged_score / len(test_indices)
dice_max2 = dice_max2_score / len(test_indices)
hm_iou = hm_iou_score / len(test_indices)

print(f"ged_score: {ged}, dice_max_score2: {dice_max2}, hm_iou_score: {hm_iou}")
logging.info(f"ged_score: {ged}, dice_max_score2: {dice_max2}, hm_iou_score: {hm_iou}")