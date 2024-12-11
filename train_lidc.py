import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from lidc_dataset import LIDC_IDRI, RandomGenerator
from utils import l2_regularisation, kl_divergence, calculate_dice_loss, calculate_sigmoid_focal_loss
from torchvision import transforms
from tensorboardX import SummaryWriter
from asam import ASAM
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Training settings')
parser.add_argument('--device', type=str, default='cuda:3', help='Device to use for training')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
parser.add_argument('--max_epoch', type=int, default=50, help='Maximum number of epochs')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimizer')
parser.add_argument('--save_path', type=str, default='checkpoint/combined_weights.pth', help='Path to save combined weights')
args = parser.parse_args()

# Ensure save directory exists
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

# Set device
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# Define MaskWeights class
class MaskWeights(nn.Module):
    def __init__(self):
        super(MaskWeights, self).__init__()
        self.weights = nn.Parameter(torch.ones(5, 1, requires_grad=True) / 6)

# Initialize network and mask weights
net = ASAM().to(device)
mask_weights = MaskWeights().to(device)
mask_weights.train()

# Load dataset
db = LIDC_IDRI(dataset_location='LIDC/data/', transform=transforms.Compose([
    RandomGenerator(output_size=[128, 128])
]))
dataset_size = len(db)
indices = list(range(dataset_size))
train_split = int(np.floor(0.6 * dataset_size))
validation_split = int(np.floor(0.8 * dataset_size))

train_indices = indices[:train_split]
validation_indices = indices[train_split:validation_split]
test_indices = indices[validation_split:]

train_dataset = Subset(db, train_indices)
validation_dataset = Subset(db, validation_indices)
test_dataset = Subset(db, test_indices)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"Total dataset size: {dataset_size}")
print(f"Training set size: {len(train_indices)}")
print(f"Validation set size: {len(validation_indices)}")
print(f"Test set size: {len(test_indices)}")

# Initialize TensorBoard writer and optimizer
writer = SummaryWriter('tf-logs/train_onestage')
optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

combined_weights = {}

# Training loop
for epoch_num in range(1, args.max_epoch + 1):
    net.train()
    loss_epoch = 0.0
    segloss_epoch = 0.0
    print(f"Epoch {epoch_num}")

    for i_batch, sampled_batch in enumerate(train_loader):
        image_batch, label_batch = sampled_batch['image'].to(device), sampled_batch['label'].to(device)
        image_batch_oc = sampled_batch['image_oc'].to(device)
        box1024_batch = sampled_batch['box_1024'].to(device)
        boxshift_batch = sampled_batch['box_shift'].to(device)

        outputs = net(image_batch, image_batch_oc, box1024_batch, boxshift_batch, label_batch, device)
        output_masks = outputs['masks']
        logits_high = output_masks.to(device)
        weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0).to(device)
        logits_high = logits_high * weights.unsqueeze(-1)
        logits_high_res = logits_high.sum(1).unsqueeze(1)

        kl1 = torch.mean(kl_divergence(net.posterior_box_latent_space, net.prior_box_latent_space))
        kl2 = torch.mean(kl_divergence(net.posterior_object_latent_space, net.prior_object_latent_space))
        cel_loss = nn.CrossEntropyLoss()(logits_high, label_batch.long())
        reg_loss = sum(l2_regularisation(layer) for layer in [
            net.prior_box, net.posterior_box, net.fcomb_box.layers,
            net.prior_object, net.posterior_object, net.fcomb_object.layers
        ])
        gt_mask = label_batch.unsqueeze(1)
        dice_loss = calculate_dice_loss(logits_high_res, gt_mask.long())
        focal_loss = calculate_sigmoid_focal_loss(logits_high_res, gt_mask.float())
        seg_loss = cel_loss + dice_loss + focal_loss
        loss = seg_loss + 1e-5 * reg_loss + kl1 + kl2

        segloss_epoch += seg_loss.item()
        loss_epoch += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Segmentation loss: {segloss_epoch / len(train_loader)}")
    print(f"Total loss: {loss_epoch / len(train_loader)}")

    # Save model weights every 10 epochs
    if epoch_num % 10 == 0:
        combined_weights[f'epoch_{epoch_num}'] = {
            'model_state_dict': {k: v.cpu() for k, v in net.state_dict().items()},
            'mask_weights': weights.cpu()
        }
        print(f"Saved weights for epoch {epoch_num}")

# Save all weights to file
torch.save(combined_weights, args.save_path)
print(f"Combined weights saved to {args.save_path}.")
