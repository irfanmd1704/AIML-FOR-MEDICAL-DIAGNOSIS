import torch
import torch.nn.functional as F

# Sample synthetic data: predicted and ground truth masks
def generate_dummy_data(size=(1, 1, 256, 256)):
    torch.manual_seed(0)
    prediction = torch.sigmoid(torch.randn(size))  # simulated model output
    target = torch.randint(0, 2, size).float()     # ground truth
    return prediction, target

# Binary Cross Entropy
def bce_loss(pred, target):
    return F.binary_cross_entropy(pred, target)

# Dice Loss
def dice_loss(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# Jaccard Loss (IoU)
def jaccard_loss(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return 1 - (intersection + smooth) / (union + smooth)

# Tversky Loss
def tversky_loss(pred, target, alpha=0.5, beta=0.5, smooth=1e-6):
    TP = (pred * target).sum()
    FP = ((1 - target) * pred).sum()
    FN = (target * (1 - pred)).sum()
    return 1 - (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

# Main: Compare losses
if __name__ == "__main__":
    pred, target = generate_dummy_data()

    print("Binary Cross Entropy Loss:", bce_loss(pred, target).item())
    print("Dice Loss:", dice_loss(pred, target).item())
    print("Jaccard Loss (IoU Loss):", jaccard_loss(pred, target).item())
    print("Tversky Loss:", tversky_loss(pred, target).item())
