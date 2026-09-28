import json
from datetime import datetime
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from typing import *
import re

from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from attack_setting import Data
from dataset import Dataset_YOUTUBE, Dataset_SACO


RELEASE_ROOT = Path(__file__).resolve().parent
DATA_BASE = Path(os.environ.get("SAM3_UAP_DATA_ROOT", RELEASE_ROOT / "data"))
DATA_ROOT1 = DATA_BASE / "dataset" / "YOUTUBE" / "train" / "JPEGImages"
DATA_ROOT2 = DATA_BASE / "sam3_data" / "media" / "saco_sav" / "JPEGImages_24fps"

def choose_dataset(args = None):
    print(f">> choose dataset from {args.train_dataset}")

    if args.train_dataset == 'YOUTUBE':

        video_dirs = [video_dir for video_dir in DATA_ROOT1.iterdir() if video_dir.is_dir()]
        video_dirs.sort(key=lambda x: x.name, reverse=True)
        num_samples = len(video_dirs) if args.limit_img == -1 else min(max(args.limit_img, 0), len(video_dirs))
        video_dirs = random.sample(video_dirs, num_samples)

        video_sample_ids = {}
        start_frames = {}
        for video_dir in video_dirs:
            if video_dir.is_dir():
                frames = [f"{video_dir.name}/{fp.stem}" for fp in video_dir.iterdir() if fp.is_file()]
                frames.sort()
                if frames:
                    first_frame_name = os.path.basename(frames[0].split('/')[-1])
                    try:
                        start_frame_idx = int(first_frame_name.lstrip('0') or '0')
                        start_frames[video_dir.name] = start_frame_idx
                    except ValueError as e:
                        print(f"Error processing first frame of {video_dir.name}: {e}")
                        start_frames[video_dir.name] = None
                if args.limit_frames > 0 and len(frames) >= args.limit_frames:
                    step = max(1, len(frames) // args.limit_frames)
                    selected_frames = frames[::step][:args.limit_frames]
                else:
                    print(f"Warning: Video {video_dir.name} has less than {args.limit_frames} frames. All frames will be used.")
                    selected_frames = frames
                selected_frames = selected_frames[::-1]
                video_sample_ids[video_dir.name] = selected_frames
        sample_ids = [item for sublist in video_sample_ids.values() for item in sublist]
        json_path = str(DATA_BASE / "dataset" / "YOUTUBE" / "train" / "Annotations")
        custom_dataset = Dataset_YOUTUBE(sample_ids, DATA_ROOT1, json_path, args=args, start_frames=start_frames)

    elif args.train_dataset == 'SACO':
        json_path = DATA_BASE / "sam3_data" / "annotation" / "saco_veval_sav_test.json"

        with open(json_path, "r") as f:
            saco_ann = json.load(f)

        videos = {v["id"]: v for v in saco_ann["videos"]}

        valid_frame_ids_from_json = set()

        for ann in saco_ann["annotations"]:
            video_id = ann["video_id"]
            video = videos[video_id]
            video_name = video["video_name"]
            file_names = video["file_names"]

            segs = ann["segmentations"]

            for frame_idx, seg in enumerate(segs):
                if seg is None:
                    continue
                if frame_idx >= len(file_names):
                    continue

                frame_rel_path = file_names[frame_idx]
                video_dir, frame_jpg = frame_rel_path.split("/")
                frame_stem = Path(frame_jpg).stem
                sample_id = f"{video_dir}/{frame_stem}"
                valid_frame_ids_from_json.add(sample_id)

        video_dirs = [video_dir for video_dir in DATA_ROOT2.iterdir() if video_dir.is_dir()]
        video_dirs.sort(key=lambda x: x.name, reverse=True)

        if args.limit_img == -1:
            target_num_videos = len(video_dirs)
        else:
            target_num_videos = min(max(args.limit_img, 0), len(video_dirs))

        video_dirs = list(video_dirs)
        random.shuffle(video_dirs)

        video_sample_ids = {}
        start_frames = {}

        for video_dir in video_dirs:
            if len(video_sample_ids) >= target_num_videos:
                break

            if not video_dir.is_dir():
                continue

            frames = [f"{video_dir.name}/{fp.stem}" for fp in video_dir.iterdir() if fp.is_file()]
            frames.sort()

            if frames:
                first_frame_name = os.path.basename(frames[0].split('/')[-1])
                try:
                    start_frame_idx = int(first_frame_name.lstrip('0') or '0')
                    start_frames[video_dir.name] = start_frame_idx
                except ValueError as e:
                    print(f"Error processing first frame of {video_dir.name}: {e}")
                    start_frames[video_dir.name] = None

            if args.limit_frames > 0 and len(frames) >= args.limit_frames:
                step = 4
                selected_frames = frames[::step][:args.limit_frames]
            else:
                print(
                    f"[SACO] Warning: Video {video_dir.name} has less than {args.limit_frames} frames. "
                    f"All frames will be used."
                )
                selected_frames = frames

            selected_frames = selected_frames[::-1]

            valid_frames = []
            for sample_id in selected_frames:
                if sample_id in valid_frame_ids_from_json:
                    valid_frames.append(sample_id)

            if not valid_frames:
                print(f"[SACO] Warning: Video {video_dir.name} has no valid annotated frames in SACO, skipping.")
                continue

            video_sample_ids[video_dir.name] = valid_frames

        if len(video_sample_ids) < target_num_videos:
            print(
                f"[SACO] Warning: Requested {target_num_videos} videos, "
                f"but only found {len(video_sample_ids)} videos with valid annotated frames."
            )

        sample_ids = [item for sublist in video_sample_ids.values() for item in sublist]

        custom_dataset = Dataset_SACO(sample_ids, DATA_ROOT2, json_path, args=args, start_frames=start_frames)
    return custom_dataset

def get_frame_index(img_ID):
    base_name = os.path.splitext(os.path.basename(img_ID))[0]

    try:
        if base_name.isdigit() and len(base_name) == 5:
            return int(base_name)
        else:
            raise ValueError(f"Unexpected frame name format: {base_name}")
    except ValueError as e:
        print(f"Error processing {img_ID}: {e}")
        raise
def save_image_only(image, video_name, frame_idx, save_dir):

    video_save_dir = os.path.join(save_dir, video_name)
    os.makedirs(video_save_dir, exist_ok=True)
    file_prefix = f'{frame_idx:04d}'
    filename = f'{file_prefix}.png'
    save_path = os.path.join(video_save_dir, filename)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_to_save = image


    cv2.imwrite(save_path, image_to_save, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"Saved image to: {save_path}")
def collate_fn(batch):
    buffer_list, P_list, sample_id,gt ,point = zip(*batch)
    return buffer_list, P_list, sample_id,gt,point
def get_video_to_indices(dataset):
    video_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        img_ID = dataset.get_img_id(idx)
        video_name = img_ID.split('/')[0]
        video_to_indices[video_name].append(idx)
    return video_to_indices
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
def get_iou(x: np.ndarray, y: np.ndarray) -> float:
  while x.ndim < y.ndim:
    x = np.expand_dims(x, 0)
  while y.ndim < x.ndim:
    y = np.expand_dims(y, 0)
  if x.shape == y.shape:
    denominator = np.sum(np.logical_or(x, y))
    return np.sum(np.logical_and(x, y)) / denominator if denominator != 0 else 0.0
  else:
    min_shape = tuple(min(a, b) for a, b in zip(x.shape, y.shape))
    x = x[tuple(slice(0, s) for s in min_shape)]
    y = y[tuple(slice(0, s) for s in min_shape)]
    denominator = np.sum(np.logical_or(x, y))
    return np.sum(np.logical_and(x, y)) / denominator if denominator != 0 else 0
def get_iou_auto(x:Union[Data, List[Data]], y:Data) -> float:
  if isinstance(x, list):
    iou = max([get_iou(m, y) for m in x])
  else:
    iou = get_iou(x, y)
  return iou
def make_print_to_file(path='./'):
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "w", encoding='utf8', )
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):

            try:
                self.terminal.flush()
            except Exception:
                pass
            try:
                self.log.flush()
            except Exception:
                pass
    fileName = datetime.now().strftime('%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)
    return fileName
def process_videos(video_root_dir, output_dir, mask_gt_dict, start_P_dict, predictor, category,video_range=None, skipped_frames=None, args=None):

    total_iou = 0
    iou_count = 0
    miou_values = []

    if skipped_frames is None:
        skipped_frames = set()

    print(f"Processing {category} samples...")

    video_names = sorted(os.listdir(video_root_dir))


    for video_name in video_names:
        video_dir = os.path.join(video_root_dir, video_name)

        if not os.path.isdir(video_dir):
            continue


        frame_names = [
            p for p in os.listdir(video_dir)
            if p.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        inference_state = predictor.init_state(video_path=video_dir)
        predictor.clear_all_points_in_video(inference_state)

        if video_name in start_P_dict:

            if args.test_prompts == 'pt':
                points = start_P_dict[video_name]
                points = np.array(points, dtype=np.float32)
                print(f"🎯 Loaded stored start_P for {video_name} as points: {points}")
                labels = np.array([1], np.int32)

                width = 1008
                height = 1008
                ann_obj_id = 1
                ann_frame_idx = 0
                rel_points = [[x / width, y / height] for x, y in points]
                points_tensor = torch.tensor(rel_points, dtype=torch.float32)
                points_labels_tensor = torch.tensor(labels, dtype=torch.int32)

                _, out_obj_ids, low_res_masks, video_res_masks = predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=points_tensor,
                    labels=points_labels_tensor,
                    clear_old_points=False,
                )

            elif args.test_prompts == 'bx':
                box = start_P_dict[video_name]

                box = np.asarray(start_P_dict[video_name], dtype=np.float32)

                print(f"🎯 Loaded stored start_P for {video_name} as box: {box}")

                width = 1008
                height = 1008
                ann_obj_id = 1
                ann_frame_idx = 0
                xmin, ymin, xmax, ymax = box

                rel_box = np.array([[xmin / width, ymin / height, xmax / width, ymax / height]], dtype=np.float32)

                rel_box = np.array(rel_box, dtype=np.float32)

                _, out_obj_ids, low_res_masks, video_res_masks = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    box=rel_box,
                )

        else:
            print(f"⚠️ {video_name} is not in start_P_dict, skipping this video")
            continue

        video_segments = {}
        max_frame_num_to_track = max(1, len(frame_names))

        for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=False,
            propagate_preflight=True,
        ):

            video_segments[frame_idx] = {
                out_obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        vis_frame_stride = 1
        for out_frame_idx in tqdm(range(0, len(frame_names), vis_frame_stride)):
            fig, ax = plt.subplots(figsize=(6, 4))
            plt.title(f"frame {out_frame_idx}")

            image_path = os.path.join(video_dir, frame_names[out_frame_idx])
            image = np.array(Image.open(image_path))
            ax.imshow(image)

            if (video_name, out_frame_idx) in mask_gt_dict:
                mask_gt = mask_gt_dict[(video_name, out_frame_idx)]
            else:
                print(f"Warning: No ground truth mask found for {video_name}, frame {out_frame_idx}")
                continue

            frame_iou_values = []
            frame_segments = video_segments.get(out_frame_idx)
            if frame_segments is None:
                print(
                    f"Warning: No propagated mask found for {video_name}, frame {out_frame_idx}; "
                    f"tracked {len(video_segments)}/{len(frame_names)} frames"
                )
                if category == "clean":
                    skipped_frames.add((video_name, out_frame_idx))
                continue


            for out_obj_id, out_mask in frame_segments.items():

                if not np.any(mask_gt > 0):
                    print(f"⚠️ {video_name}, frame {out_frame_idx}: mask_gt is all zeros, skipping IoU computation")
                    continue

                iou_img = get_iou_auto(out_mask, mask_gt)


                if category == "clean" and iou_img < 0.0:
                    print(f"🚫 Skipping frame {out_frame_idx} of {video_name} due to low IoU ({iou_img:.4f})")
                    skipped_frames.add((video_name, out_frame_idx))
                    continue

                if category == "adversarial" and (video_name, out_frame_idx) in skipped_frames:
                    print(f"🚫 Skipping adversarial IoU for {video_name}, frame {out_frame_idx}, as clean IoU was < 0.15")
                    continue

                total_iou += iou_img
                iou_count += 1
                frame_iou_values.append(iou_img)
                print(f"IoU for {category} {video_name}, frame {out_frame_idx}, object {out_obj_id}: {iou_img:.4f}")

            if frame_iou_values:
                miou_values.append(np.mean(frame_iou_values))

            if args is not None and getattr(args, "save_eval_figures", False):
                os.makedirs(output_dir, exist_ok=True)
                save_prefix = "clean" if category == "clean" else "adv"
                save_path = os.path.join(output_dir, f"{video_name}_{save_prefix}_frame_{out_frame_idx:04d}.jpg")
                plt.savefig(save_path)
                plt.close(fig)
    avg_iou = total_iou / iou_count if iou_count > 0 else 0
    miou = np.mean(miou_values) if miou_values else 0

    if category == "clean":
        print(f"Average IoU for clean samples: {avg_iou:.4f}")
    else:
        print(f"Average IoU for adversarial samples: {avg_iou:.4f}")

    return miou, iou_count, skipped_frames


def process_images_text(
    video_root_dir,
    output_dir,
    mask_gt_dict,
    start_P_dict,
    processor,
    category,
    video_range=None,
    skipped_frames=None,
    args=None,
):
    total_iou = 0.0
    iou_count = 0
    miou_values = []

    if skipped_frames is None:
        skipped_frames = set()

    iou_skip_threshold = 0.15
    if args is not None and hasattr(args, "iou_skip_threshold"):
        iou_skip_threshold = float(getattr(args, "iou_skip_threshold"))

    print(f"Processing {category} samples...")

    video_names = sorted(os.listdir(video_root_dir))

    if video_range is not None:
        start_idx, end_idx = video_range
        video_names = video_names[start_idx:end_idx]

    for video_name in video_names:
        video_dir = os.path.join(video_root_dir, video_name)
        if not os.path.isdir(video_dir):
            continue

        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        if video_name in start_P_dict:
            if args is not None and getattr(args, "test_prompts", None) == "pt":
                text_prompt = start_P_dict[video_name]
                if not isinstance(text_prompt, str):
                    text_prompt = str(text_prompt)
                print(f"🎯 Loaded stored start_P for {video_name} as text prompt: {text_prompt}")
            else:
                text_prompt = getattr(args, "text_prompt", "horse in the scene")
                print(f"ℹ️ Using args.text_prompt or the default prompt for {video_name}: {text_prompt}")
        else:
            print(f"⚠️ {video_name} is not in start_P_dict, skipping this video")
            continue

        print(f"▶ Processing video: {video_name}, {len(frame_names)} frames")

        vis_frame_stride = 1

        for idx in tqdm(range(0, len(frame_names), vis_frame_stride)):
            image_path = os.path.join(video_dir, frame_names[idx])
            image = Image.open(image_path).convert("RGB")

            stem = os.path.splitext(frame_names[idx])[0]
            try:
                frame_id = int(stem)
            except ValueError:
                print(f"Warning: frame name not numeric: {frame_names[idx]}")
                continue

            if category == "adversarial" and (video_name, frame_id) in skipped_frames:
                print(
                    f"🚫 Skipping adversarial IoU for {video_name}, frame {frame_id}, "
                    f"as clean frame was skipped"
                )
                continue

            inference_state = processor.set_image(image)
            output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)

            masks = output["masks"]
            if isinstance(masks, torch.Tensor):
                masks = masks.detach().cpu().numpy()

            frame_segments = {
                obj_id: (masks[obj_id] > 0.0)
                for obj_id in range(masks.shape[0])
            }

            if (video_name, frame_id) in mask_gt_dict:
                mask_gt = mask_gt_dict[(video_name, frame_id)]
            else:
                print(f"Warning: No ground truth mask found for {video_name}, frame {frame_id}")
                continue

            if not np.any(mask_gt > 0):
                print(f"⚠️ {video_name}, frame {frame_id}: mask_gt is all zeros, skipping IoU computation")
                continue

            best_iou = None

            for out_obj_id, out_mask in frame_segments.items():
                iou_img = get_iou_auto(out_mask, mask_gt)
                if best_iou is None or iou_img > best_iou:
                    best_iou = iou_img



            if best_iou is None:
                continue

            if category == "clean" and best_iou < iou_skip_threshold:
                print(
                    f"🚫 Skipping frame {frame_id} of {video_name} "
                    f"due to low IoU ({best_iou:.4f} < {iou_skip_threshold})"
                )
                skipped_frames.add((video_name, frame_id))
                continue

            total_iou += float(best_iou)
            iou_count += 1
            miou_values.append(float(best_iou))

            print(
                f"[FRAME] IoU for {category} {video_name}, frame {frame_id}: {best_iou:.4f}"
            )

            if args is not None and getattr(args, "save_eval_figures", False):
                os.makedirs(output_dir, exist_ok=True)
                save_prefix = "clean" if category == "clean" else "adv"
                save_path = os.path.join(
                    output_dir,
                    f"{video_name}_{save_prefix}_frame_{frame_id:04d}.jpg",
                )
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.imshow(image)
                ax.set_title(f"{video_name} frame {frame_id} ({save_prefix})")
                plt.savefig(save_path)
                plt.close(fig)

    avg_iou = total_iou / iou_count if iou_count > 0 else 0.0
    miou = float(np.mean(miou_values)) if miou_values else 0.0

    if category == "clean":
        print(f"Average IoU for clean samples (frame-level): {avg_iou:.4f} (count={iou_count})")
    else:
        print(f"Average IoU for adversarial samples (frame-level): {avg_iou:.4f} (count={iou_count})")

    return miou, iou_count, skipped_frames

def process_videos_text(
    video_root_dir,
    output_dir,
    mask_gt_dict,
    start_P_dict,
    video_predictor,
    category,
    video_range=None,
    skipped_frames=None,
    args=None,
):
    total_iou = 0.0
    iou_count = 0
    miou_values = []
    frame_iou_dict = {}
    frame_prob_dict = {}
    prob_below_thresh_count = 0


    adv_no_mask_counted_zero_frames = set()

    if skipped_frames is None:
        skipped_frames = set()

    print(f"Processing {category} samples...")

    def _natural_video_key(name: str):
        nums = re.findall(r"\d+", name)
        if nums:
            return (0, int(nums[-1]), name)
        return (1, name)

    video_names = [
        n for n in os.listdir(video_root_dir)
        if os.path.isdir(os.path.join(video_root_dir, n))
    ]
    video_names.sort(key=_natural_video_key)

    if video_range is not None:
        start_idx, end_idx = video_range
        video_names = video_names[start_idx:end_idx]

    if video_predictor is None:
        raise ValueError(
            "video_predictor must be provided and support handle_request / handle_stream_request."
        )


    def propagate_in_video(predictor, session_id):

        outputs_per_frame_raw = {}
        for response in predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=session_id)
        ):
            outputs_per_frame_raw[response["frame_index"]] = response["outputs"]
        return outputs_per_frame_raw

    def _safe_close_session(predictor, session_id):
        if session_id is None:
            return
        try:
            predictor.handle_request({"type": "close_session", "session_id": session_id})
        except Exception as e:
            print(f"Warning: failed to close session {session_id}: {e}")

    def _is_missing_conditioning_error(exc):
        return "No points are provided; please add points first" in str(exc)

    def _run_video_session_with_retry(predictor, video_dir, text_prompt, video_name):
        last_exc = None
        for attempt in range(2):
            session_id = None
            try:
                resp = predictor.handle_request(
                    request=dict(type="start_session", resource_path=video_dir)
                )
                session_id = resp["session_id"]

                _ = predictor.handle_request(
                    request=dict(type="add_prompt", session_id=session_id, frame_index=0, text=text_prompt)
                )

                outputs_per_frame_raw = propagate_in_video(predictor, session_id)
                return session_id, outputs_per_frame_raw, attempt
            except RuntimeError as exc:
                _safe_close_session(predictor, session_id)
                if not _is_missing_conditioning_error(exc):
                    raise
                last_exc = exc
                print(
                    f"[session-retry] {video_name}: SAM3.1 multiplex conditioning state invalid on attempt {attempt + 1}/2: {exc}"
                )
                torch.cuda.empty_cache()
            except Exception:
                _safe_close_session(predictor, session_id)
                raise
        raise last_exc


    def _to_numpy(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _extract_frame_array(arr_all, frame_idx, num_frames=None):

        if arr_all is None:
            return None

        if isinstance(arr_all, (list, tuple)):
            if frame_idx < 0 or frame_idx >= len(arr_all):
                return None
            return _to_numpy(arr_all[frame_idx])

        arr = _to_numpy(arr_all)
        if arr is None:
            return None

        if num_frames is not None and arr.ndim >= 1:

            if arr.shape[0] == num_frames:
                return arr[frame_idx] if 0 <= frame_idx < arr.shape[0] else None

            if arr.ndim >= 2 and arr.shape[1] == num_frames:
                return arr[:, frame_idx, ...] if 0 <= frame_idx < arr.shape[1] else None

        return arr

    def _extract_frame_output(outputs, frame_idx, num_frames=None):

        if outputs is None:
            return None, None, None, None

        if isinstance(outputs, (list, tuple)):
            if frame_idx < 0 or frame_idx >= len(outputs):
                return None, None, None, None
            return _extract_frame_output(outputs[frame_idx], 0, num_frames=1)

        if not isinstance(outputs, dict):
            return None, None, None, None

        masks_all = outputs.get("out_binary_masks", None)
        probs_all = outputs.get("out_probs", None)
        obj_ids_all = outputs.get("out_obj_ids", None)
        boxes_all = outputs.get("out_boxes_xywh", None)

        masks = _extract_frame_array(masks_all, frame_idx, num_frames=num_frames)
        probs = _extract_frame_array(probs_all, frame_idx, num_frames=num_frames)
        obj_ids = _extract_frame_array(obj_ids_all, frame_idx, num_frames=num_frames)
        boxes = _extract_frame_array(boxes_all, frame_idx, num_frames=num_frames)

        if masks is not None:
            masks = np.asarray(masks)
            if masks.ndim == 2:
                masks = masks[None, ...]
            elif masks.ndim != 3:
                masks = None
            else:
                if masks.dtype != np.bool_:
                    masks = masks > 0

        if probs is not None:
            probs = np.asarray(probs).reshape(-1)
        if obj_ids is not None:
            obj_ids = np.asarray(obj_ids).reshape(-1)
        if boxes is not None:
            boxes = np.asarray(boxes)

        return masks, probs, obj_ids, boxes


    for video_name in video_names:
        video_dir = os.path.join(video_root_dir, video_name)
        if not os.path.isdir(video_dir):
            continue

        frame_names = [
            p for p in os.listdir(video_dir)
            if p.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        try:
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        except ValueError:
            frame_names.sort()


        if video_name in start_P_dict:
            if args is not None and getattr(args, "test_prompts", None) == "text":
                text_prompt = start_P_dict[video_name]
                if not isinstance(text_prompt, str):
                    text_prompt = str(text_prompt)
                print(f"🎯 Loaded stored start_P for {video_name} as text prompt: {text_prompt}")
            else:
                text_prompt = getattr(args, "text_prompt", "horse in the scene")
                print(f"ℹ️ Using args.text_prompt or the default prompt for {video_name}: {text_prompt}")
        else:
            print(f"⚠️ {video_name} is not in start_P_dict, skipping this video")
            continue

        print(f"▶ Processing video: {video_name}, {len(frame_names)} frames")


        session_id = None
        try:
            session_id, outputs_per_frame_raw, retry_idx = _run_video_session_with_retry(
                video_predictor, video_dir, text_prompt, video_name
            )
            if retry_idx > 0:
                print(f"[session-retry] {video_name}: recovered after retry")
        except RuntimeError as e:
            if not _is_missing_conditioning_error(e):
                raise

            print(f"[session-skip] {video_name}: failed after retry due to invalid SAM3.1 conditioning state")

            if category == "clean":
                for out_frame_idx in range(0, len(frame_names), 1):
                    vf = (video_name, out_frame_idx)
                    if vf in mask_gt_dict:
                        skipped_frames.add(vf)
                torch.cuda.empty_cache()
                continue

            for out_frame_idx in range(0, len(frame_names), 1):
                vf = (video_name, out_frame_idx)
                if vf not in mask_gt_dict or vf in skipped_frames:
                    continue

                mask_gt = mask_gt_dict[vf]
                if not np.any(mask_gt > 0):
                    continue

                iou_img = 0.0
                total_iou += iou_img
                iou_count += 1
                miou_values.append(iou_img)
                frame_iou_dict[vf] = iou_img
                frame_prob_dict[vf] = None
                adv_no_mask_counted_zero_frames.add(vf)

            torch.cuda.empty_cache()
            continue


        for out_frame_idx in tqdm(range(0, len(frame_names), 1)):
            vf = (video_name, out_frame_idx)


            if vf not in mask_gt_dict:
                print(f"[{category}] {video_name}, frame {out_frame_idx}: GT missing, skip")
                continue

            mask_gt = mask_gt_dict[vf]
            if not np.any(mask_gt > 0):
                print(f"⚠️ {video_name}, frame {out_frame_idx}: mask_gt is all zeros, skipping IoU computation")
                continue


            if category == "adversarial" and vf in skipped_frames:
                print(
                    f"🚫 Skipping adversarial IoU for {video_name}, frame {out_frame_idx}, "
                    f"as it was skipped in clean"
                )
                continue

            outputs_this_frame = outputs_per_frame_raw.get(out_frame_idx, None)


            masks, probs, obj_ids, boxes = _extract_frame_output(
                outputs_this_frame, frame_idx=0, num_frames=None
            )
            if masks is None and isinstance(outputs_this_frame, dict):

                masks2, probs2, obj_ids2, boxes2 = _extract_frame_output(
                    outputs_this_frame, frame_idx=out_frame_idx, num_frames=len(frame_names)
                )
                if masks2 is not None:
                    masks, probs, obj_ids, boxes = masks2, probs2, obj_ids2, boxes2


            if masks is None or masks.shape[0] == 0:
                if category == "clean":

                    skipped_frames.add(vf)
                    print(f"[clean] {video_name}, frame {out_frame_idx}: no masks -> skipped (NOT counted)")
                    continue


                iou_img = 0.0
                total_iou += iou_img
                iou_count += 1
                miou_values.append(iou_img)
                frame_iou_dict[vf] = iou_img
                frame_prob_dict[vf] = None
                adv_no_mask_counted_zero_frames.add(vf)

                print(
                    f"[adversarial] {video_name}, frame {out_frame_idx}: no masks -> "
                    f"COUNTED as IoU=0.0 (clean had valid frame)"
                )
                continue

            if probs is not None and len(probs) == masks.shape[0]:
                best_i = int(np.argmax(probs))
            else:
                best_i = 0

            best_mask = masks[best_i].astype(bool)
            iou_img = float(get_iou_auto(best_mask, mask_gt))


            low_thr = getattr(args, "skip_iou_thr", None)
            if category == "clean" and low_thr is not None and iou_img < float(low_thr):
                skipped_frames.add(vf)
                print(
                    f"🚫 Skipping frame {out_frame_idx} of {video_name} "
                    f"due to low IoU ({iou_img:.4f} < {float(low_thr):.4f})"
                )
                continue

            total_iou += iou_img
            iou_count += 1
            miou_values.append(iou_img)
            frame_iou_dict[vf] = iou_img


            top_prob = None
            if probs is not None and len(probs) > 0:
                top_prob = float(probs[best_i])
                if top_prob < 0.5:
                    prob_below_thresh_count += 1
            frame_prob_dict[vf] = top_prob

            oid_str = ""
            if obj_ids is not None and best_i < len(obj_ids):
                oid_str = f", obj_id {int(obj_ids[best_i])}"
            print(f"IoU for {category} {video_name}, frame {out_frame_idx}, top1{oid_str}: {iou_img:.4f}")

        _safe_close_session(video_predictor, session_id)
        torch.cuda.empty_cache()

    avg_iou = total_iou / iou_count if iou_count > 0 else 0.0
    miou = float(np.mean(miou_values)) if miou_values else 0.0

    if category == "clean":
        print(f"Average IoU for clean samples: {avg_iou:.4f}")
        return miou, iou_count, skipped_frames, frame_iou_dict, frame_prob_dict, prob_below_thresh_count

    print(f"Average IoU for adversarial samples: {avg_iou:.4f}")
    print(
        f"[adversarial] Counted-as-zero due to no masks (clean valid): "
        f"{len(adv_no_mask_counted_zero_frames)} frames"
    )
    return (
        miou,
        iou_count,
        adv_no_mask_counted_zero_frames,
        frame_iou_dict,
        frame_prob_dict,
        prob_below_thresh_count,
    )