import numpy as np
import torch
from medpy import metric
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            print(f'Focal loss alpha={alpha}, will assign alpha values for each class')
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(f'Focal loss alpha={alpha}, will shrink the impact in background')
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            self.alpha[1:] = 1 - alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds, labels):
        self.alpha = self.alpha.to(preds.device)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        B, H, W = labels.shape
        assert B * H * W == preds.shape[0]
        assert preds.shape[-1] == self.num_classes
        preds_logsoft = F.log_softmax(preds, dim=1)
        preds_softmax = torch.exp(preds_logsoft)

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0

def l2_regularisation(m):
    l2_reg = None
    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

def iou_score_cal(prediction, groundtruth):
    prediction = prediction.detach().cpu().numpy()
    groundtruth = groundtruth.detach().cpu().numpy()
    intersection = np.logical_and(groundtruth, prediction)
    union = np.logical_or(groundtruth, prediction)
    if np.sum(union) == 0:
        return 1
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def mask_IoU(prediction, groundtruth):
    prediction = prediction.detach().cpu().numpy()
    groundtruth = groundtruth.detach().cpu().numpy()
    intersection = np.logical_and(groundtruth, prediction)
    union = np.logical_or(groundtruth, prediction)
    if np.sum(union) == 0:
        return 1
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def generalized_energy_distance_iou(predictions, masks):
    n = predictions.shape[0]
    m = masks.shape[0]
    d1 = d2 = d3 = 0
    for i in range(n):
        for j in range(m):
            d1 += (1 - mask_IoU(predictions[i], masks[j]))

    for i in range(n):
        for j in range(n):
            d2 += (1 - mask_IoU(predictions[i], predictions[j]))

    for i in range(m):
        for j in range(m):
            d3 += (1 - mask_IoU(masks[i], masks[j]))

    d1 = d1 * (2 / (n * m))
    d2 = d2 * (1 / (n * n))
    d3 = d3 * (1 / (m * m))

    ed = d1 - d2 - d3
    scores = mask_IoU(predictions[0], masks[0])

    return ed, scores

def dice_score_cal(pred, targs):
    pred = (pred > 0).float()
    intersection = (pred * targs).sum()
    union = pred.sum() + targs.sum()
    if union == 0:
        return 1.0
    dice_score = 2. * intersection / union
    return dice_score

def dice_coef_cal(output, target):
    smooth = 1e-5
    output = output.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)

def iou(pred, true):
    pred_bool = pred.bool().detach().cpu()
    true_bool = true.bool().detach().cpu()
    intersection = (pred_bool & true_bool).float().sum()
    union = (pred_bool | true_bool).float().sum()
    if union == 0 and intersection == 0:
        return 1
    else:
        return intersection / union

def hm_iou_cal(preds, trues):
    num_preds = len(preds)
    num_trues = len(trues)
    cost_matrix = torch.zeros((num_preds, num_trues))
    for i, pred in enumerate(preds):
        for j, true in enumerate(trues):
            cost_matrix[i, j] = 1 - iou(pred, true)
    row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())
    matched_iou = [iou(preds[i], trues[j]) for i, j in zip(row_ind, col_ind)]
    avg_iou = torch.FloatTensor(matched_iou).mean().item()
    return avg_iou

def calculate_dice_loss(inputs, targets, num_masks=5):
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

def calculate_sigmoid_focal_loss(inputs, targets, num_masks=5, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks

def dice_max_cal1(pred_eval, label_four):
    dice_max = 0
    for i in range(pred_eval.shape[0]):
        dice_max_iter = 0
        for j in range(label_four.shape[0]):
            dice_score_iter = dice_score_cal(pred_eval[i].to(dtype=torch.float).squeeze().cpu().detach(), label_four[j].squeeze(0).cpu().detach())
            if j == 0:
                dice_max_iter = dice_score_iter
            else:
                if dice_score_iter > dice_max_iter:
                    dice_max_iter = dice_score_iter
        dice_max += dice_max_iter
    return dice_max / pred_eval.shape[0]

def dice_max_cal2(pred_eval, label_four):
    dice_max = -1
    for i in range(pred_eval.shape[0]):
        for j in range(label_four.shape[0]):
            dice_score_iter = dice_score_cal(pred_eval[i].to(dtype=torch.float).squeeze().cpu().detach(), label_four[j].squeeze(0).cpu().detach())
            if dice_score_iter > dice_max:
                dice_max = dice_score_iter
    return dice_max

def dice_avg_cal(pred_list, label_four):
    dice_all = 0
    pred_stack = torch.stack(pred_list)
    pred_avg = torch.mean(pred_stack, dim=0)
    pred_avg = (pred_avg > 0).cpu().detach()
    pred_avg = torch.where(pred_avg, torch.tensor(1), torch.tensor(0))

    for i in range(label_four.shape[0]):
        dice_score_iter = dice_score_cal(pred_avg.to(dtype=torch.float).squeeze().cpu().detach(), label_four[i].squeeze(0).cpu().detach())
        dice_all += dice_score_iter
    return dice_all / label_four.shape[0]

def kl_divergence(posterior_latent_space, prior_latent_space, analytic=True, calculate_posterior=False, z_posterior=None):
    if analytic:
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
    else:
        if calculate_posterior:
            z_posterior = posterior_latent_space.rsample()
        log_posterior_prob = posterior_latent_space.log_prob(z_posterior)
        log_prior_prob = prior_latent_space.log_prob(z_posterior)
        kl_div = log_posterior_prob - log_prior_prob
    return kl_div

def show_mask(mask, ax, color):
    h, w = mask.shape[-2:]
    color = np.array(color + [0.5])
    mask_image = np.zeros((h, w, 4))
    for i in range(3):
        mask_image[:, :, i] = mask.squeeze() * color[i]
    mask_image[:, :, 3] = (mask.squeeze() > 0) * color[3]
    ax.imshow(mask_image)

def show_box(box, ax, color):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor='none', lw=2))

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        truncated_normal_(m.bias, mean=0, std=0.001)

def init_weights_orthogonal_normal(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)