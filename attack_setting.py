import os
import random
from argparse import ArgumentParser, Namespace
from collections import deque
from typing import Tuple, Callable, Union, List

import numpy as np
import torch
import math

from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.nn import Module
from torch.autograd import Function
import pywt
from torchvision.transforms import Resize, Normalize

import torch.nn.functional as F
from numpy import ndarray
from numpy.typing import NDArray

from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3.model.sam3_tracker_base import Sam3TrackerBase
from sam3.model.sam3_video_base import Sam3VideoBase
from torchvision.transforms import v2

from sam3.model.utils.sam1_utils import SAM2Transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

try:
    from moviepy.editor import ImageSequenceClip
    HAS_MOVIEPY = True
except ImportError:
    print('>> [warn] missing lib "moviepy", will not generate adv pred step by step')
    HAS_MOVIEPY = False
npimg_u8  = NDArray[np.uint8]
npimg_u16 = NDArray[np.uint16]
npimg_f32 = NDArray[np.float32]
npimg_b1  = NDArray[np.bool_]
npimg     = Union[npimg_u8, npimg_f32, npimg_b1]
Data      = Union[ndarray, Tensor]
Size      = Tuple[int, int]
Point     = Tuple[int, int]
Prompts   = Tuple[ndarray, ndarray, None, None]
CAM_METH = ['GradCAM', 'HiResCAM', 'ScoreCAM', 'GradCAMPlusPlus', 'AblationCAM', 'XGradCAM', 'EigenCAM', 'FullGrad']
LIM = ['', 'edge', 'smap', 'cam', 'tgt']
SAM_MASK_THRESH = 0.0
class SamForwarder(nn.Module):
    def __init__(self, sam: Sam3TrackerBase):
        super().__init__()
        self.model = sam
        self.device = 'cuda:0'
        self.mask_threshold = 0.0
        self.max_hole_area = 256.0
        self.max_sprinkle_area = 0.0
        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=self.mask_threshold,
            max_hole_area=self.max_hole_area,
            max_sprinkle_area=self.max_sprinkle_area,
        )
        self._bb_feat_sizes = [
            (288, 288),
            (144, 144),
            (72, 72),
        ]
        self.NO_OBJ_SCORE = -1024.0

        # Keep normalization tensors as buffers to avoid recreating CUDA tensors.
        self.register_buffer("pixel_mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

        self.add_all_frames_to_correct_as_cond = False
        self.max_cond_frames_in_attn = -1
        self.fill_hole_area = 0

        self._features = {}
        self._orig_hw: List[Tuple[int, int]] = []
        self._is_image_set = False

        assert SAM_MASK_THRESH == self.mask_threshold, \
            f'sam.mask_threshold ({sam.mask_threshold}) != SAM_MASK_THRESH ({SAM_MASK_THRESH})'

    # Cache cleanup helpers.
    def clear_cache(self):
        """
        Clear cached features held by SamForwarder.

        This releases intermediate references only and does not modify model
        parameters.
        """
        self._features = {}
        self._is_image_set = False

    def reset(self):
        """
        Reset state before processing a new video or sequence.
        """
        self.clear_cache()
        if hasattr(self.model, "reset_state"):
            self.model.reset_state()
        elif hasattr(self.model, "reset_memory"):
            self.model.reset_memory()

    def norm_image(self, x: Tensor) -> Tensor:
        return (x.to(self.pixel_mean.device) - self.pixel_mean) / self.pixel_std

    def denorm_image(self, x: Tensor) -> Tensor:
        return x.to(self.pixel_mean.device) * self.pixel_std + self.pixel_mean

    def resize_image(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        return F.pad(x, (0, self.canvas_size - w, 0, self.canvas_size - h))

    def unresize_image(self, x: Tensor) -> Tensor:
        INTERP_MODE = 'bilinear'
        align_corners = None if INTERP_MODE == 'nearest' else False
        x = F.interpolate(x, (self.canvas_size, self.canvas_size), mode=INTERP_MODE, align_corners=align_corners)
        x = x[..., :self.input_size[0], :self.input_size[1]]
        return F.interpolate(x, self.original_size, mode=INTERP_MODE, align_corners=align_corners)

    def transform_image(self, im: npimg_u8, is_edge: bool = False) -> Tensor:
        if isinstance(im, np.ndarray):
            self._orig_hw = [im.shape[:2]]
        x = self._transforms(im)
        X = x[None, ...].to(self.device)
        return X

    def transform_prompts(self, point_coords: np.ndarray = None, point_labels: np.ndarray = None,
                          box: np.ndarray = None,
                          mask_logits: np.ndarray = None, normalize_coords=True) -> Tuple[
        Tensor, Tensor, Tensor, Tensor]:
        img_idx = -1
        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert (
                    point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device
            )
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )  # Bx2x2
        if mask_logits is not None:
            mask_input = torch.as_tensor(
                mask_logits, dtype=torch.float, device=self.device
            )
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return unnorm_coords, labels, unnorm_box, mask_input

    def forward(self, image: Tensor, point_coords: Tensor = None, point_labels: Tensor = None, boxes: Tensor = None,
                mask_input: Tensor = None, multi_mask: bool = False) -> Tensor:  # Tuple[Tensor]:

        img_idx: int = -1
        backbone_out = self.model.forward_image(image)
        (
            _,
            vision_feats,
            _,
            _,
        ) = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
                    feat.permute(1, 2, 0).view(1, -1, *feat_size)
                    for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
                ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True


        concat_points = (point_coords, point_labels) if point_coords is not None else None
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_input,
        )
        batched_mode = (concat_points is not None and concat_points[0].shape[0] > 1)  # multi object prediction
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multi_mask,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )
        masks = self._transforms.postprocess_masks(
            low_res_masks, self._orig_hw[img_idx]
        )
        print(masks.shape)

        return masks

    def forward_logits(
        self,
        image: Tensor,
        point_coords: Tensor = None,
        point_labels: Tensor = None,
        boxes: Tensor = None,
        mask_input: Tensor = None,
        multi_mask: bool = False,
        upscale: bool = True,
    ) -> Tensor:
        """Return low-resolution mask logits, optionally upsampled to the input size."""

        img_idx: int = -1
        backbone_out = self.model.forward_image(image)
        (_, vision_feats, _, _) = self.model._prepare_backbone_features(backbone_out)
        vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]

        concat_points = (point_coords, point_labels) if point_coords is not None else None
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_input,
        )
        batched_mode = concat_points is not None and concat_points[0].shape[0] > 1

        high_res_features = [feat_level[img_idx].unsqueeze(0) for feat_level in feats[:-1]]
        low_res_masks, _, _, _ = self.model.sam_mask_decoder(
            image_embeddings=feats[-1][img_idx].unsqueeze(0),
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multi_mask,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        if not upscale:
            return low_res_masks

        target_hw = self._orig_hw[img_idx] if self._orig_hw else image.shape[-2:]
        return F.interpolate(low_res_masks, size=target_hw, mode='bilinear', align_corners=False)


FwdPack = Tuple[SamForwarder, Prompts, Callable]

def seed_everything(seed: int):
    print('>> global seed:', seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(">> seed everything...")


def print_memory_usage(step_description):
    id = 1
    allocated = torch.cuda.memory_allocated(id)
    reserved = torch.cuda.memory_reserved(id)
    print(f"[{step_description}] Allocated: {allocated / 1024 ** 2:.2f} MB, Reserved: {reserved / 1024 ** 2:.2f} MB")

def _parse_point(coord: Union[str, None], size: Size, prompt_num: int = 1) -> List[Point]:
    if coord:
        point = list(reversed([float(e) for e in coord.split(',')]))
        for i, e in enumerate(point):
            if e < 1.0:
                point[i] = e * size[i]
            point[i] = int(point[i])
        return [tuple(point)]
    else:
        grid_size = math.ceil(math.sqrt(prompt_num))
        width, height = size
        x_step = width // grid_size
        y_step = height // grid_size

        grid_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = np.random.randint(i * x_step, (i + 1) * x_step)
                y = np.random.randint(j * y_step, (j + 1) * y_step)
                grid_points.append((x, y))
        return grid_points


def make_multi_prompts(points: Union[str, np.ndarray], img_size: Size, prompt_num: int = 1) -> Prompts:
    if isinstance(points, str) or points is None:
        grid_points = _parse_point(points, img_size, prompt_num)
        coords = np.array(grid_points, dtype=np.int32)
        labels = np.ones((coords.shape[0],), dtype=np.int32)
    else:
        coords = np.asarray(points, dtype=np.int32)
        assert coords.ndim == 2 and coords.shape[1] == 2, "Points must have shape (N, 2)"

        if coords.shape[0] < prompt_num:
            grid_points = _parse_point(None, img_size, prompt_num)
            grid_points = np.array(grid_points, dtype=np.int32)
            coords = np.vstack([coords, grid_points[:prompt_num - coords.shape[0]]])
        elif coords.shape[0] > prompt_num:
            coords = coords[:prompt_num, :]

        labels = np.ones((coords.shape[0],), dtype=np.int32)

    return (coords, labels, None, None)
def _parse_box(coord: Union[str, None], size: Size, prompt_num: int = 1) -> List[Tuple[Point, Point]]:
    if coord:
        box = [float(e) for e in coord.split(',')]
        x1, y1, x2, y2 = box
        for i, e in enumerate([x1, y1, x2, y2]):
            if e < 1.0:
                if i % 2 == 0:
                    box[i] = e * size[0]
                else:
                    box[i] = e * size[1]
            box[i] = int(box[i])
        return [(tuple(box[:2]), tuple(box[2:]))]
    else:
        grid_size = math.ceil(math.sqrt(prompt_num))
        width, height = size
        x_step = width // grid_size
        y_step = height // grid_size

        grid_boxes = []
        for i in range(grid_size):
            for j in range(grid_size):
                x1 = i * x_step
                y1 = j * y_step
                x2 = x1 + x_step
                y2 = y1 + y_step
                grid_boxes.append(((x1, y1), (x2, y2)))
        return grid_boxes


def make_multi_prompts_box(points: Union[str, np.ndarray], img_size: Size, prompt_num: int = 1) -> Prompts:
    if isinstance(points, str) or points is None:
        grid_boxes = _parse_box(points, img_size, prompt_num)

        box_coords = []
        for (x1, y1), (x2, y2) in grid_boxes:
            box = np.array([x1, y1, x2, y2], dtype=np.int32)
            box = box.reshape((1, 1, 4))
            box_coords.append(box)

        all_boxes = np.vstack(box_coords).reshape(-1, 4)
        min_x1 = np.min(all_boxes[:, 0])
        min_y1 = np.min(all_boxes[:, 1])
        max_x2 = np.max(all_boxes[:, 2])
        max_y2 = np.max(all_boxes[:, 3])
        combined_box = np.array([min_x1, min_y1, max_x2, max_y2], dtype=np.int32).reshape(1, 1, 4)
        boxes = combined_box

        print("1")
    else:
        boxes = np.asarray(points, dtype=np.int32)
        assert boxes.ndim == 2 and boxes.shape[1] == 4, "Points must have shape (N, 4)"

        if boxes.shape[0] < prompt_num:
            grid_boxes = _parse_box(None, img_size, prompt_num)
            grid_boxes = np.array([[x1, y1, x2, y2] for (x1, y1), (x2, y2) in grid_boxes], dtype=np.int32)
            boxes = np.vstack([boxes, grid_boxes[:prompt_num - boxes.shape[0]]])
            print("2")
        elif boxes.shape[0] > prompt_num:
            boxes = boxes[:prompt_num, :]
            print("3")

    return (None, None, boxes, None)
def _parse_point1(coord:str, size:Size) -> Point:
  if coord:
    point = list(reversed([float(e) for e in coord.split(',')]))
    for i, e in enumerate(point):
      if e < 1.0: point[i] = e * size[i]
      point[i] = int(point[i])
  else:
    point = [random.randrange(sz) for sz in size]
  return point
def make_prompts(point: Union[str, np.ndarray], img_size: tuple) -> Prompts:
    if isinstance(point, str) or point is None:
        point = _parse_point1(point, img_size)
        coords = np.expand_dims(np.asarray(point, dtype=np.int32), axis=0)
    else:
        coords = point
    labels = np.asarray([1], dtype=np.int32)
    return (coords, labels, None, None)

def make_prompts_randombox(box: Union[str, ndarray],img_size: tuple) -> Prompts:
  boxes = []
  y1, x1 = np.random.randint(0, img_size[0]), np.random.randint(0, img_size[1])
  y2, x2 = np.random.randint(y1, img_size[0]), np.random.randint(x1, img_size[1])
  boxes.append((y1, x1, y2, x2))
  box_array = np.expand_dims(np.asarray(boxes, dtype=np.int32), axis=0)
  labels = np.asarray([1], dtype=np.int32)
  return (None, labels, box_array, None)

def make_prompts_box(box: Union[str, np.ndarray], img_size: tuple) -> Prompts:
    box_array = np.asarray(box, dtype=np.int32)
    labels = np.asarray([1], dtype=np.int32)
    return (None, labels, box_array, None)
