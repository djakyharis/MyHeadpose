import os
import cv2
import logging
import argparse
import numpy as np
import math

import torch
from torchvision import transforms

from ptflops import get_model_complexity_info

from models import get_model
from utils.datasets import AFLW2000
from utils.general import compute_euler_angles_from_rotation_matrices

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def parse_args():
    parser = argparse.ArgumentParser(description='Head pose estimation evaluation.')
    parser.add_argument('--data', type=str, default='data/AFLW2000/', help='Directory path for data.')
    parser.add_argument("--network", type=str, default="resnet18",
                        help="Network architecture: resnet18/34/50, mobilenetv2, mobilenetv3_small")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers for data loading.")
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size.')
    parser.add_argument('--weights', type=str, default='', help='Path to model weight for evaluation.')
    return parser.parse_args()


# ================================
# 1. Geodesic Rotation Error
# ================================
def geodesic_error(R_pred, R_gt):
    R_diff = torch.matmul(R_pred.transpose(1, 2), R_gt)
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]

    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1 + 1e-6, 1 - 1e-6)

    theta = torch.acos(cos_theta)
    return theta * 180 / math.pi  # convert to degree


@torch.no_grad()
def evaluate(params, model, data_loader, device):
    model.eval()

    total = 0
    yaw_error = pitch_error = roll_error = 0.0

    geodesic_errors = []

    for images, R_gt, cont_labels, name in data_loader:
        images = images.to(device)
        R_gt = R_gt.to(device)
        total += cont_labels.size(0)

        # GT Euler angles
        p_gt_deg = cont_labels[:, 0].float() * 180 / np.pi
        y_gt_deg = cont_labels[:, 1].float() * 180 / np.pi
        r_gt_deg = cont_labels[:, 2].float() * 180 / np.pi

        # Model prediction
        R_pred = model(images)
        euler = compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi

        p_pred_deg = euler[:, 0].cpu()
        y_pred_deg = euler[:, 1].cpu()
        r_pred_deg = euler[:, 2].cpu()

        # ==========================
        # 2. Geodesic rotation error
        # ==========================
        ge_err = geodesic_error(R_pred, R_gt)
        geodesic_errors.append(torch.sum(ge_err).item())

        # Euler MAE (existing)
        pitch_error += torch.sum(torch.abs(p_gt_deg - p_pred_deg))
        yaw_error += torch.sum(torch.abs(y_gt_deg - y_pred_deg))
        roll_error += torch.sum(torch.abs(r_gt_deg - r_pred_deg))

    # Print Euler angle MAE
    logging.info(
        f'Yaw: {yaw_error / total:.4f} '
        f'Pitch: {pitch_error / total:.4f} '
        f'Roll: {roll_error / total:.4f} '
        f'MAE: {(yaw_error + pitch_error + roll_error) / (total * 3):.4f}'
    )

    # Print Geodesic Error
    geo_mean = sum(geodesic_errors) / total
    logging.info(f'Geodesic Rotation Error: {geo_mean:.4f}°')


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_dataset = AFLW2000(params.data, transform=eval_transform)
    data_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        pin_memory=True
    )
    logging.info('Loading test data.')

    # Load model
    model = get_model(params.network, num_classes=6, pretrained=False)
    if os.path.exists(params.weights):
        model.load_state_dict(torch.load(params.weights, map_location=device))
    else:
        raise ValueError(f"Model weight not found at {params.weights}")
    model.to(device)

    # ==========================
    # 3. PARAMETER COUNT
    # ==========================
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f'Total Parameters: {total_params:,}')

    # ==========================
    # 4. FLOPs estimation
    # ==========================
    macs, params_flops = get_model_complexity_info(
        model, (3, 224, 224), as_strings=True, print_per_layer_stat=False
    )
    logging.info(f'MACs: {macs}')
    logging.info(f'FLOPs (~= 2×MACs): {macs}')

    evaluate(params=params, model=model, data_loader=data_loader, device=device)


if __name__ == '__main__':
    args = parse_args()
    main(args)
