import numpy as np
import torch
import cv2
import os

from transform_3d import (
    PhotoMetricDistortionMultiViewImage,
    NormalizeMultiviewImage,
    PadMultiViewImage,
)

class MultiViewImagePreprocessor:
    def __init__(
        self,
        img_norm_cfg=None,
        size_divisor=32,
        photometric_cfg=None,
    ):
        """
        img_norm_cfg: dict with mean, std, to_rgb
        size_divisor: padding divisor
        photometric_cfg: dict for PhotoMetricDistortionMultiViewImage
        """
        if img_norm_cfg is None:
            img_norm_cfg = dict(
                mean=[103.530, 116.280, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False,
            )

        if photometric_cfg is None:
            photometric_cfg = dict(
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18,
            )

        self.photo_aug = PhotoMetricDistortionMultiViewImage(**photometric_cfg)
        self.normalize = NormalizeMultiviewImage(**img_norm_cfg)
        self.pad = PadMultiViewImage(size_divisor=size_divisor)

    def prepare_batch(self, images_list, augment=True):
        """
        images_list: list of (H, W, 3) BGR numpy arrays, uint8 or float32
        augment: apply photometric distortion if True
        returns: torch.Tensor (N, 3, H', W') float32
        """
        # Ensure float32 in [0, 255]
        imgs = [img.astype(np.float32) if img.dtype != np.float32 else img.copy() for img in images_list]
        if imgs[0].max() <= 1.0:
            imgs = [img * 255.0 for img in imgs]

        results = {"img": imgs}

        if augment:
            results = self.photo_aug(results)

        results = self.normalize(results)
        results = self.pad(results)

        # (N, H, W, 3) -> (N, 3, H, W)
        out = np.stack(results["img"], axis=0)
        out = torch.from_numpy(out).permute(0, 3, 1, 2)
        return out


if __name__ == "__main__":
    path_to_images = "images"
    num_cams = 4

    image_paths = [
        os.path.join(path_to_images, f"image_{i}.png")
        for i in range(num_cams)
    ]

    images = [cv2.imread(p) for p in image_paths]  # BGR

    preprocessor = MultiViewImagePreprocessor()

    batch_train = preprocessor.prepare_batch(images, augment=True)
    batch_eval = preprocessor.prepare_batch(images, augment=False)

    print(batch_train.shape)  # (num_cams, 3, H', W')
    print(batch_eval.shape)   # (num_cams, 3, H', W')
