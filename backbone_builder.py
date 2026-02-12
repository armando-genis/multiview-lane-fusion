import os
import torch
import cv2
from mmengine import Config
from mmdet.registry import MODELS
from lane_fusion import MultiViewImagePreprocessor

if __name__ == "__main__":

    backbone_cfg = dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(
            type='DCNv2',
            deform_groups=1,
            fallback_on_stride=False
        ),
        stage_with_dcn=(False, False, True, True),
    )

    backbone = MODELS.build(backbone_cfg)
    print(backbone)
    print("Backbone built successfully")


    path_to_images = "images"
    num_cams = 4

    image_paths = [
        os.path.join(path_to_images, f"image_{i}.png")
        for i in range(num_cams)
    ]

    images = [cv2.imread(p) for p in image_paths]  # BGR

    preprocessor = MultiViewImagePreprocessor()

    batch_train = preprocessor.prepare_batch(images, augment=True)

    backbone.eval()
    with torch.no_grad():
        feats = backbone(batch_train)  # tuple of 3 tensors: C3, C4, C5

    # feats[0]: (4, 512, H/8, W/8), feats[1]: (4, 1024, ...), feats[2]: (4, 2048, ...)
    for i, f in enumerate(feats):
        print(f"Level {i}: {f.shape}")

