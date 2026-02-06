import sys
import numpy as np
import torch
import cv2
import os
from transform_3d import PhotoMetricDistortionMultiViewImage, NormalizeMultiviewImage, PadMultiViewImage

IMG_NORM_CFG = dict(
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0],
    to_rgb=False,
)

photo_aug = PhotoMetricDistortionMultiViewImage(
    brightness_delta=32,
    contrast_range=(0.5, 1.5),
    saturation_range=(0.5, 1.5),
    hue_delta=18,
)
normalize = NormalizeMultiviewImage(**IMG_NORM_CFG)
pad = PadMultiViewImage(size_divisor=32)

def prepare_batch(images_list, augment=True):
    """
    images_list: list of (H, W, 3) BGR numpy, uint8 or float32.
    augment: if True, apply PhotoMetricDistortionMultiViewImage (training).
    Returns: tensor (N, 3, H', W') float32.
    """
    # Pipeline expects list of float32 in [0, 255]
    imgs = [img.astype(np.float32) if img.dtype != np.float32 else img.copy() for img in images_list]
    if imgs[0].max() <= 1.0:
        imgs = [img * 255.0 for img in imgs]

    results = {"img": imgs}

    if augment:
        results = photo_aug(results)
    results = normalize(results)
    results = pad(results)

    # Stack to tensor (N, 3, H, W)
    out = np.stack(results["img"], axis=0)
    out = torch.from_numpy(out).permute(0, 3, 1, 2)
    return out

if __name__ == "__main__":

    path_to_images = "images"
    num_cams = 4
    images = [os.path.join(path_to_images, f"image_{i}.png") for i in range(num_cams)]
    six_imgs = [cv2.imread(img) for img in images]  # BGR
    batch_train = prepare_batch(six_imgs, augment=True)   # (num_cams, 3, H', W')
    batch_eval = prepare_batch(six_imgs, augment=False)

    print(batch_train.shape)  # (num_cams, 3, H', W')
    print(batch_eval.shape)   # (num_cams, 3, H', W')


