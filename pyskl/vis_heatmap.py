import moviepy.editor as mpy
import matplotlib.cm as cm
import numpy as np
import cv2
from .pipelines import Compose
limb_pipeline = [
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(type='GeneratePoseTarget', with_kp=False, with_limb=True)
]

keypoint_pipeline = [
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False)
]

def get_pseudo_heatmap(anno, flag='keypoint'):
    assert flag in ['keypoint', 'limb']
    pipeline = Compose(keypoint_pipeline if flag == 'keypoint' else limb_pipeline)
    return pipeline(anno)['imgs']

def vis_heatmaps(heatmaps, channel=-1, ratio=(8,8)):
    # if channel is -1, draw all keypoints / limbs on the same map
    heatmaps = [x.transpose(1, 2, 0) for x in heatmaps]
    h, w, _ = heatmaps[0].shape
    newh, neww = int(h * ratio[0]), int(w * ratio[1])
    
    if channel == -1:
        heatmaps = [np.max(x, axis=-1) for x in heatmaps]
    cmap = cm.viridis
    heatmaps = [(cmap(x)[..., :3] * 255).astype(np.uint8) for x in heatmaps]
    heatmaps = [cv2.resize(x, (neww, newh)) for x in heatmaps]
    return heatmaps

def to_pseudo_heatmap(anno, flag='keypoint'):
    return np.array(get_pseudo_heatmap(anno, flag)).transpose(1, 0, 2, 3)
#     limb_mapvis = vis_heatmaps(limb_heatmap, ratio=ratio)

def to_heatmap(anno, flag='keypoint', ratio=(8,8)):
    limb_heatmap = get_pseudo_heatmap(anno, flag)
    limb_mapvis = vis_heatmaps(limb_heatmap, ratio=ratio)
    # limb_mapvis = [add_label(f, gym_categories[anno['label']]) for f in limb_mapvis]
    # if show_video:
    #     vid = mpy.ImageSequenceClip(limb_mapvis, fps=24)
    #     vid.ipython_display()

    return limb_mapvis