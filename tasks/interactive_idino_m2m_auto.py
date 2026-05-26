# --------------------------------------------------------
# Semantic-SAM: Segment and Recognize Anything at Any Granularity
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Hao Zhang (hzhangcx@connect.ust.hk)
# --------------------------------------------------------

import torch
import numpy as np
from torchvision import transforms
from utils.visualizer import Visualizer
from typing import Tuple
from PIL import Image, ImageOps      
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt
import cv2
import io
from .automatic_mask_generator import SemanticSamAutomaticMaskGenerator
from skimage import measure
metadata = MetadataCatalog.get('coco_2017_train_panoptic')

# def interactive_infer_image(model, image,level,all_classes,all_parts, thresh,text_size,hole_scale,island_scale,semantic, refimg=None, reftxt=None, audio_pth=None, video_pth=None):
#     t = []
#     t.append(transforms.Resize(int(text_size), interpolation=Image.BICUBIC))
#     transform1 = transforms.Compose(t)
#     image_ori = transform1(image)

#     image_ori = np.asarray(image_ori)
#     images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()

#     mask_generator = SemanticSamAutomaticMaskGenerator(model,points_per_side=32,
#             pred_iou_thresh=0.88,
#             stability_score_thresh=0.92,
#             min_mask_region_area=10,
#             level=level,
#         )

#     outputs = mask_generator.generate(images)

#     fig=plt.figure(figsize=(10, 10))
#     plt.imshow(image_ori)
#     show_anns(outputs)
#     fig.canvas.draw()
#     im=Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
#     return im
def interactive_infer_image(
    model, image, level, all_classes, all_parts,
    thresh, text_size, hole_scale, island_scale,
    semantic, refimg=None, reftxt=None, audio_pth=None, video_pth=None
):
    # 1) EXIF 보정
    image = ImageOps.exif_transpose(image).convert("RGB")
    orig_w, orig_h = image.size

    # 2) 짧은 변 = text_size 로 리사이즈
    resize_tf  = transforms.Resize(int(text_size), interpolation=Image.BICUBIC)
    image_rs   = resize_tf(image)
    rs_w, rs_h = image_rs.size

    # 3) Semantic-SAM 마스크
    img_t = torch.from_numpy(np.asarray(image_rs)).permute(2,0,1).cuda()
    mask_gen = SemanticSamAutomaticMaskGenerator(
        model, points_per_side=32, pred_iou_thresh=0.88,
        stability_score_thresh=0.92, min_mask_region_area=10, level=level)
    outputs = mask_gen.generate(img_t)

    # 4) 시각화용 이미지
    fig = plt.figure(figsize=(10,10)); plt.imshow(image_rs); show_anns(outputs)
    fig.canvas.draw()
    vis_img = Image.frombytes("RGB", fig.canvas.get_width_height(),
                              fig.canvas.tostring_rgb())
    plt.close(fig)

    # 5) ✨ 패딩 계산 (SAM은 항상 TEXT_SIZE×TEXT_SIZE 로 패딩)
    if orig_w >= orig_h:            # landscape → 위·아래 패딩
        pad_x, pad_y = 0, (text_size - rs_h) // 2
    else:                           # portrait → 좌·우 패딩
        pad_x, pad_y = (text_size - rs_w) // 2, 0

    sx, sy = orig_w / rs_w, orig_h / rs_h   # 스케일 팩터

    # 6) SVG path 변환
    segments = []
    for i, ann in enumerate(outputs):
        contours = measure.find_contours(ann["segmentation"].astype(float), 0.5)
        if not contours: continue
        # 가장 큰 contour 하나만 사용
        cnt = np.fliplr(contours[0])                         # (row,col) → (x,y)
        path = "M " + " L ".join(
            f"{(x-pad_x)*sx:.2f},{(y-pad_y)*sy:.2f}" for x,y in cnt
        ) + " Z"
        segments.append({
            "id": i,
            "label": ann.get("label", f"segment {i}"),
            "path": path,
        })

    print(">>> RETURNING segments and image")
    return segments, vis_img

def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
) -> Tuple[np.ndarray, bool]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))