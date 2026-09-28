import json
import os
import re
from pathlib import Path
from typing import Tuple

from pycocotools import mask as mask_utils

import random
from attack_setting import *
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import numpy as np
from sam3.model_builder import build_sam3_image_model, build_sam3_video_model
from sam3.model.sam3_image_processor import Sam3Processor

def calculate_center(gt):
    coords = np.column_stack(np.where(gt > 0))

    x_center = np.mean(coords[:, 1])
    y_center = np.mean(coords[:, 0])

    return x_center, y_center
def calculate_bounding_box(gt: np.ndarray) -> Tuple[int, int, int, int]:
    coords = np.column_stack(np.where(gt > 0))

    if len(coords) == 0:
        return 0, 0, 0, 0

    x_min = np.min(coords[:, 1])
    y_min = np.min(coords[:, 0])
    x_max = np.max(coords[:, 1])
    y_max = np.max(coords[:, 0])

    return x_min, y_min, x_max, y_max

def _build_prompt_and_annotation(gt, image, sam_fwder, train_prompts):

    mode = train_prompts if train_prompts is not None else "pt"

    if mode == "pt":
        x_center, y_center = calculate_center(gt)
        prompt_ann = np.array([[x_center, y_center]], dtype=np.float32)
        prompts = make_prompts(prompt_ann, image)
        P = sam_fwder.transform_prompts(*prompts)
        return P, prompt_ann

    if mode == "bx":
        x_min, y_min, x_max, y_max = calculate_bounding_box(gt)
        prompt_ann = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
        prompts = make_prompts_box(prompt_ann, image)
        P = sam_fwder.transform_prompts(*prompts)
        return P, prompt_ann

    if mode == "text":

        x_center, y_center = calculate_center(gt)
        point_ann = np.array([[x_center, y_center]], dtype=np.float32)
        prompts = make_prompts(point_ann, image)
        P = sam_fwder.transform_prompts(*prompts)
        return P, "object"

    raise ValueError(f"Unsupported train_prompts: {mode}")


def get_mask_for_color(mask, target_color):

    target_color = np.array(target_color, dtype=np.uint8)
    color_mask = np.all(mask == target_color, axis=-1)

    gt_region = np.zeros_like(mask)

    gt_region[color_mask] = target_color

    return color_mask, gt_region
def retain_specific_color_with_single_channel(gt_region, target_color):

    target_color = np.array(target_color, dtype=np.uint8)

    color_mask = np.all(gt_region == target_color, axis=-1)

    single_channel_mask = np.where(color_mask, 255, 0).astype(np.uint8)

    return single_channel_mask


class Dataset_YOUTUBE(Dataset):

    def __init__(self, sample_ids, data_root, json_path,target_size=(1008, 1008), do_shuffle=False,args = None,start_frames = None):
        self.sample_ids = sample_ids
        self.data_root = data_root
        self.json_path = json_path
        self.target_size = target_size
        self.do_shuffle = do_shuffle
        self.args = args
        self.sam = build_sam3_video_model()
        self.start_frames = start_frames or {}
        combined_data = list(zip(self.sample_ids, self._load_data()))
        self.sample_ids, self.data = zip(*combined_data[::-1])

    def _load_data(self):
        data = []
        failed_samples = []
        for sample_id in self.sample_ids:

            image_path = self.data_root / f"{sample_id}.jpg"
            gt_paths = Path(self.json_path) / f"{sample_id}.png"
            try:
                image = Image.open(image_path).convert("RGB")
                image = np.array(image.resize(self.target_size[::-1], Image.BILINEAR))


                pil_image = Image.open(gt_paths).convert('RGBA')

                mask = np.array(pil_image)
                if mask is None or mask.size == 0:
                    raise FileNotFoundError(f"Could not read image from path: {gt_paths}")



                target_color = (236, 95, 103, 255)
                color_mask, gt_region = get_mask_for_color(mask, target_color)

                single_channel_mask = retain_specific_color_with_single_channel(gt_region, target_color)
                single_mask = Image.fromarray(single_channel_mask)

                gt = np.array(single_mask.resize(self.target_size[::-1], Image.NEAREST))


                sam_fwder = SamForwarder(self.sam)
                X = sam_fwder.transform_image(image)

                P, prompt_ann = _build_prompt_and_annotation(
                    gt, image, sam_fwder, getattr(self.args, "train_prompts", None)
                )

                data.append((image, P, sample_id, gt, prompt_ann))
                print(f"from YOUTUBE Processing sample {sample_id}...")
            except Exception as e:
                print(f"Error loading data for sample {sample_id}: {str(e)}")
                failed_samples.append(sample_id)
                continue
        return data
    def __len__(self):
        return len(self.sample_ids)
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        return self.data[idx]
    def get_start_frame_idx(self, video_name):
        """Return the first frame index recorded for the given video."""
        return self.start_frames.get(video_name, None)

    def get_img_id(self, idx):

        try:
            if isinstance(idx, int):
                if idx < 0 or idx >= len(self.sample_ids):
                    raise IndexError(f"Index {idx} is out of range. Total samples: {len(self.sample_ids)}")
                sample_id = self.sample_ids[idx]
            elif isinstance(idx, str):
                sample_id = idx
            else:
                raise TypeError(f"Expected idx to be an integer or string, but got {type(idx)}")

            match = re.search(r"([a-zA-Z0-9]+)\/\d+", sample_id)
            if match:
                img_id = match.group(1)
                return img_id
            else:
                raise ValueError(f"Could not extract image ID from sample_id: {sample_id}")
        except Exception as e:
            print(f"Error extracting image ID from idx {idx}: {str(e)}")
            return None


class Dataset_SACO(Dataset):
    def __init__(
        self,
        sample_ids,
        data_root,
        json_path,
        target_size=(1008, 1008),
        do_shuffle=False,
        args=None,
        start_frames=None,
    ):
        self.sample_ids = sample_ids
        self.data_root = data_root
        self.json_path = json_path
        self.target_size = target_size
        self.do_shuffle = do_shuffle
        self.args = args
        self.sam = build_sam3_video_model()
        self.start_frames = start_frames or {}

        with open(self.json_path, "r") as f:
            ann = json.load(f)

        self.videos_by_id = {v["id"]: v for v in ann["videos"]}
        self.videos_by_name = {v["video_name"]: v for v in ann["videos"]}

        self.annotations = ann["annotations"]

        self.sample_meta = {}

        for ann_item in self.annotations:
            video_id = ann_item["video_id"]
            video = self.videos_by_id[video_id]
            video_name = video["video_name"]
            file_names = video["file_names"]

            segs = ann_item["segmentations"]

            for frame_idx, seg in enumerate(segs):
                if seg is None:
                    continue
                if frame_idx >= len(file_names):
                    continue

                frame_rel_path = file_names[frame_idx]
                video_dir, frame_jpg = frame_rel_path.split("/")
                frame_stem = Path(frame_jpg).stem
                sample_id = f"{video_dir}/{frame_stem}"

                meta = self.sample_meta.setdefault(
                    sample_id,
                    {
                        "video": video,
                        "frame_idx": frame_idx,
                        "anns": [],
                    },
                )
                meta["anns"].append(ann_item)

        if self.do_shuffle:
            from random import shuffle
            shuffle(self.sample_ids)

        combined_data = list(zip(self.sample_ids, self._load_data()))
        if combined_data:
            self.sample_ids, self.data = zip(*combined_data[::-1])
            self.sample_ids = list(self.sample_ids)
            self.data = list(self.data)
        else:
            self.sample_ids, self.data = [], []

    def _load_data(self):
        data = []
        failed_samples = []

        for sample_id in self.sample_ids:
            try:
                if sample_id not in self.sample_meta:
                    print(f"[SACO] sample_id {sample_id} not found in JSON annotations, skip.")
                    failed_samples.append(sample_id)
                    continue

                meta = self.sample_meta[sample_id]
                video = meta["video"]
                frame_idx = meta["frame_idx"]
                anns = meta["anns"]

                ann_item = anns[0]
                noun_phrase = ann_item["noun_phrase"]
                segs = ann_item["segmentations"]
                seg = segs[frame_idx]

                image_path = self.data_root / f"{sample_id}.jpg"
                if not image_path.exists():
                    raise FileNotFoundError(f"Image not found: {image_path}")

                image = Image.open(image_path).convert("RGB")
                image = np.array(
                    image.resize(self.target_size[::-1], Image.BILINEAR)
                )
                print(image.shape)

                gt = None
                if seg is not None:
                    height = ann_item["height"]
                    width = ann_item["width"]

                    rle = seg
                    if isinstance(rle.get("counts"), list):
                        rle = mask_utils.frPyObjects([rle], height, width)[0]

                    mask = mask_utils.decode(rle)
                    mask = (mask.astype(np.uint8) * 255)
                    mask = Image.fromarray(mask)
                    gt = np.array(
                        mask.resize(self.target_size[::-1], Image.NEAREST)
                    )

                if self.args is not None and self.args.train_prompts == "text":
                    print("text")

                    prompt_ann = str(noun_phrase)
                    P = prompt_ann
                elif self.args.train_prompts == "pt":
                    print("pt")
                    sam_fwder = SamForwarder(self.sam)
                    X = sam_fwder.transform_image(image)
                    x_center, y_center = calculate_center(gt)
                    prompt_ann = np.array([[x_center, y_center]], dtype=np.float32)
                    prompts = make_prompts(prompt_ann, image)
                    P = sam_fwder.transform_prompts(*prompts)
                elif self.args.train_prompts == "bx":
                    print("bx")
                    sam_fwder = SamForwarder(self.sam)
                    X = sam_fwder.transform_image(image)
                    x_min, y_min, x_max, y_max = calculate_bounding_box(gt)
                    prompt_ann = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
                    prompts = make_prompts_box(prompt_ann, image)
                    P = sam_fwder.transform_prompts(*prompts)
                else:
                    prompt_ann = None
                    P = None

                data.append((image, P, sample_id, gt, prompt_ann))
                print(f"[SACO] Processing sample {sample_id}...")

            except Exception as e:
                print(f"[SACO] Error loading data for sample {sample_id}: {str(e)}")
                failed_samples.append(sample_id)
                continue

        return data

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_start_frame_idx(self, video_name):
        """Return the first frame index recorded for the given video."""
        return self.start_frames.get(video_name, None)

    def get_img_id(self, idx):
        try:
            if isinstance(idx, int):
                if idx < 0 or idx >= len(self.sample_ids):
                    raise IndexError(
                        f"Index {idx} is out of range. Total samples: {len(self.sample_ids)}"
                    )
                sample_id = self.sample_ids[idx]
            elif isinstance(idx, str):
                sample_id = idx
            else:
                raise TypeError(
                    f"Expected idx to be an integer or string, but got {type(idx)}"
                )

            match = re.search(r"([a-zA-Z0-9_]+)/\d+", sample_id)
            if match:
                img_id = match.group(1)
                return img_id
            else:
                raise ValueError(
                    f"Could not extract image ID from sample_id: {sample_id}"
                )
        except Exception as e:
            print(f"Error extracting image ID from idx {idx}: {str(e)}")
            return None