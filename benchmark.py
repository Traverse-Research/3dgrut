import argparse
import torch

import os

import numpy as np
import torch
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from PIL import Image
from torchvision.transforms import ToTensor
from threedgrut.utils.logger import logger
import json
import logging
from prettytable import PrettyTable

def load_image(path, device):
    img = Image.open(path).convert("RGB")
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)  # [1, 3, H, W]
    return img_tensor

@torch.no_grad()
def evaluate_from_disk(pred_dir, gt_dir, log_file, device="cuda", compute_extra_metrics=True):
    """
    Evaluate image quality metrics between predicted and GT images on disk.

    Args:
        pred_dir (str): Path to predicted images folder.
        gt_dir (str): Path to ground truth images folder.
        device (str): 'cuda' or 'cpu'
        compute_extra_metrics (bool): Whether to compute SSIM and LPIPS.

    Returns:
        Dict of average metrics.
    """

    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(message)s',
        level=logging.INFO,
        encoding='utf-16LE'
    )
    logging.info("\Render Evaluation")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='[%(levelname)s] %(message)s',
        level=logging.INFO,
        encoding='utf-16LE'
    )

    # Set up metrics
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device) if compute_extra_metrics else None
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to("cuda") if compute_extra_metrics else None

    psnr, ssim, lpips = [], [], []

    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".png")])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".png")])

    assert len(pred_files) == len(gt_files), "Predicted and GT folders must contain the same number of images."

    for i, (pred_name, gt_name) in enumerate(zip(pred_files, gt_files)):
        pred_path = os.path.join(pred_dir, pred_name)
        gt_path = os.path.join(gt_dir, gt_name)

        pred_img = load_image(pred_path, device)
        gt_img = load_image(gt_path, device)

        # Ensure same size
        assert pred_img.shape == gt_img.shape, f"Image size mismatch: {pred_name}"

        psnr_value = psnr_metric(pred_img, gt_img).item()
        psnr.append(psnr_value)

        if compute_extra_metrics:
            ssim_value = ssim_metric(pred_img, gt_img).item()
            ssim.append(ssim_value)

            lpips_value = lpips_metric(pred_img, gt_img).item()
            lpips.append(lpips_value)

        logging.info(f"Frame {i}, PSNR: {psnr_value}")

    json_file_path = pred_dir + "/frametime.json"
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    frame_times = data.get('frame_times', [])

    if not frame_times:
        print("No frame times found in the JSON file.")

    mean_inference_time = np.mean(frame_times)

    mean_psnr = np.mean(psnr)
    std_psnr = np.std(psnr)
    
    results = {
        "mean_psnr": mean_psnr,
        "std_psnr": std_psnr,
    }

    if compute_extra_metrics:
        mean_ssim = np.mean(ssim)
        mean_lpips = np.mean(lpips)

        table = dict(
                mean_psnr=mean_psnr,
                mean_ssim=mean_ssim,
                mean_lpips=mean_lpips,
                std_psnr=std_psnr,
        )

        results["mean_ssim"] = mean_ssim
        results["mean_lpips"] = mean_lpips
    else:
        table = dict(
            mean_psnr=mean_psnr,
            std_psnr=std_psnr,
        )

    table["mean_inference_time"] = f"{'{:.2f}'.format(mean_inference_time)}" + " ms/frame"

    logger.log_table(f"Test Metrics predicted and GT images on disk", record=table)

    logtable = PrettyTable()
    logtable.field_names = list(table.keys())
    row = [f"{v:.3f}" if isinstance(v, float) else v for v in table.values()]
    logtable.add_row(row)

    for col in table.keys():
        logtable.align[col] = "l" 

    logging.info("Test Metrics predicted and GT images on disk - Step 30000 \n%s", logtable)

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True, type=str, help="Path to the predicted images")
    parser.add_argument("--gt-dir", required=True, type=str, help="Path to the GT images")
    parser.add_argument("--log-file", required=True, type=str, help="Path to the log to append")
    parser.add_argument("--compute-extra-metrics", action="store_false", help="If set, extra image metrics will not be computed [True by default]")
    args = parser.parse_args()

    evaluate_from_disk(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        log_file=args.log_file,
        device="cuda",
        compute_extra_metrics=args.compute_extra_metrics
    )
