"""Evaluate a saved universal presence-token perturbation on video datasets.

Text prompts keep the original SAM3 text evaluation path.
Point/box prompts reuse the video-style evaluation path with prompts read from the dataset.
"""

import csv
import os
import sys
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sam3_util import (
    choose_dataset,
    collate_fn,
    get_frame_index,
    get_frame_index_seg,
    make_print_to_file,
    process_videos,
    process_videos_text,
    seed_everything,
    get_video_to_indices,
)
_UAP_MODULAR_ROOT = _PROJECT_ROOT / "attack" / "uap_modular"
if str(_UAP_MODULAR_ROOT) not in sys.path:
    sys.path.insert(0, str(_UAP_MODULAR_ROOT))

from sam3.model.geometry_encoders import Prompt
from sam3.model_builder import build_sam3_video_model, build_sam3_video_predictor
from text_prompt_variants import parse_text_prompt_value
from uap_losses_logits import forward_get_toplogit


ALLOWED_TEXT_DATASETS = {"SACO"}
ALLOWED_GEOM_DATASETS = {"YOUTUBE", "MOSE", "DAVIS"}
ALLOWED_PROMPTS = {"text", "pt", "bx"}

RELEASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("SAM3_UAP_OUTPUT_ROOT", RELEASE_ROOT / "outputs" / "uapdelta"))


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Eval UAP delta on SAM3 video text benchmark")
    parser.add_argument("--delta_path", default=str(RELEASE_ROOT / "outputs" / "predlogit_uap_delta_modular_epoch15.pt"))
    parser.add_argument("--limit_img", default=100, type=int)
    parser.add_argument("--limit_frames", default=15, type=int)
    parser.add_argument("--train_dataset", default="SACO")
    parser.add_argument("--train_prompts", default="text")
    parser.add_argument("--test_prompts", default="text", choices=sorted(ALLOWED_PROMPTS))
    parser.add_argument("--checkpoints", default="sam2-t")
    parser.add_argument("--seed", default=30, type=int)
    parser.add_argument("--save", action="store_true", help="save csv summary")
    parser.add_argument("--save_img_with_mask", action="store_true")
    parser.add_argument(
        "--enable_multi_point_eval",
        action="store_true",
        help="For pt evaluation, replace the dataset start point with multiple foreground points sampled from the start-frame mask.",
    )
    parser.add_argument(
        "--multi_point_num",
        default=1,
        type=int,
        help="Number of foreground points to sample when --enable_multi_point_eval is enabled.",
    )
    parser.add_argument("--run_tag", default="", help="Optional run tag. If empty, auto timestamp is used.")
    return parser.parse_args()


def _validate_and_normalize_args(args: Namespace) -> None:
    if args.test_prompts not in ALLOWED_PROMPTS:
        raise ValueError(f"Unsupported --test_prompts={args.test_prompts}. Expected one of {sorted(ALLOWED_PROMPTS)}")

    if args.multi_point_num < 1:
        raise ValueError(f"--multi_point_num must be >= 1, got {args.multi_point_num}")

    if args.test_prompts == "text":
        if args.train_dataset not in ALLOWED_TEXT_DATASETS:
            raise ValueError(
                f"Text-prompt eval only supports datasets {sorted(ALLOWED_TEXT_DATASETS)}, got {args.train_dataset}"
            )
    else:
        if args.train_dataset not in ALLOWED_GEOM_DATASETS:
            raise ValueError(
                f"Point/box eval only supports datasets {sorted(ALLOWED_GEOM_DATASETS)}, got {args.train_dataset}"
            )

    if args.train_prompts != args.test_prompts:
        print(
            f"Overriding train_prompts from {args.train_prompts} to {args.test_prompts} "
            "so prompts are read from the dataset in the evaluation mode."
        )
    args.train_prompts = args.test_prompts

    if args.enable_multi_point_eval and args.test_prompts != "pt":
        print(
            f"[warn] --enable_multi_point_eval only applies to pt prompts, got {args.test_prompts}; disabling it."
        )
        args.enable_multi_point_eval = False


def _resolve_run_tag(args: Namespace) -> str:
    tag = str(args.run_tag).strip()
    if tag:
        return tag
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_size(img_t: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    if tuple(img_t.shape[-2:]) == target_hw:
        return img_t
    return F.interpolate(img_t.unsqueeze(0), size=target_hw, mode="bilinear", align_corners=False).squeeze(0)


def _tensor_to_uint8_hwc(img_t: torch.Tensor) -> np.ndarray:
    """Convert tensor image to uint8 HWC, supporting CHW/HWC and batch=1 variants."""
    t = img_t.detach().to(torch.float32).cpu()

    if t.ndim == 4:
        if t.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for image tensor, got shape {tuple(t.shape)}")
        t = t.squeeze(0)

    if t.ndim != 3:
        raise ValueError(f"Expected 3D image tensor after squeeze, got shape {tuple(t.shape)}")

    if t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)
    elif t.shape[-1] not in (1, 3):
        raise ValueError(f"Unsupported image tensor layout with shape {tuple(t.shape)}")

    return (t.clamp(0.0, 1.0).numpy() * 255.0).astype("uint8")


def _normalize_delta_to_chw(delta: torch.Tensor) -> torch.Tensor:
    """Normalize delta to CHW layout for stable broadcasting with CHW images."""
    d = delta.detach().to(torch.float32)

    if d.ndim == 4:
        if d.shape[0] != 1:
            raise ValueError(f"Expected delta batch size 1 for 4D tensor, got shape {tuple(d.shape)}")
        d = d.squeeze(0)

    if d.ndim == 2:
        d = d.unsqueeze(0)

    if d.ndim != 3:
        raise ValueError(f"Unsupported delta shape: {tuple(d.shape)}")

    if d.shape[0] in (1, 3):
        return d
    if d.shape[-1] in (1, 3):
        return d.permute(2, 0, 1)

    raise ValueError(f"Unsupported delta channel layout: {tuple(d.shape)}")


def _mask_to_numpy(mask: Any) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask after squeeze, got shape {tuple(arr.shape)}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Mask contains non-finite values: shape={tuple(arr.shape)}")
    return arr


def _sample_foreground_points_from_mask(mask: Any, num_points: int, context: str) -> np.ndarray:
    mask_arr = _mask_to_numpy(mask)
    foreground_yx = np.argwhere(mask_arr > 0)
    if foreground_yx.size == 0:
        raise ValueError(f"[{context}] mask_gt has no foreground pixels for multi-point sampling")

    sample_idx = np.linspace(0, len(foreground_yx) - 1, num=max(int(num_points), 1), dtype=int)
    sampled_yx = foreground_yx[sample_idx]
    return sampled_yx[:, [1, 0]].astype(np.float32)


def _expand_point_labels_if_needed(points: Any, labels: Any) -> Tuple[Any, Any]:
    if points is None or labels is None:
        return points, labels

    if isinstance(points, torch.Tensor):
        point_count = int(points.detach().cpu().reshape(-1, 2).shape[0])
    else:
        point_count = int(np.asarray(points, dtype=np.float32).reshape(-1, 2).shape[0])

    if point_count <= 1:
        return points, labels

    if isinstance(labels, torch.Tensor):
        flat_labels = labels.reshape(-1)
        if flat_labels.numel() == 1:
            return points, torch.full((point_count,), int(flat_labels.item()), dtype=labels.dtype, device=labels.device)
        return points, labels

    flat_labels = np.asarray(labels).reshape(-1)
    if flat_labels.size == 1:
        return points, np.full((point_count,), int(flat_labels[0]), dtype=np.int32)
    return points, labels


def _enable_multi_point_label_broadcast(predictor):
    if getattr(predictor, "_multi_point_label_broadcast_wrapped", False):
        return predictor

    orig_add_new_points = predictor.add_new_points
    orig_add_new_points_or_box = predictor.add_new_points_or_box

    def wrapped_add_new_points(*args, **kwargs):
        points, labels = _expand_point_labels_if_needed(kwargs.get("points"), kwargs.get("labels"))
        if points is not None:
            kwargs["points"] = points
        if labels is not None:
            kwargs["labels"] = labels
        return orig_add_new_points(*args, **kwargs)

    def wrapped_add_new_points_or_box(*args, **kwargs):
        points, labels = _expand_point_labels_if_needed(kwargs.get("points"), kwargs.get("labels"))
        if points is not None:
            kwargs["points"] = points
        if labels is not None:
            kwargs["labels"] = labels
        return orig_add_new_points_or_box(*args, **kwargs)

    predictor.add_new_points = wrapped_add_new_points
    predictor.add_new_points_or_box = wrapped_add_new_points_or_box
    predictor._multi_point_label_broadcast_wrapped = True
    return predictor


def _save_image_png(image: np.ndarray, video_name: str, frame_idx: int, save_dir: str) -> None:
    """Save image as lossless PNG to avoid compression destroying small perturbations."""
    video_save_dir = os.path.join(save_dir, video_name)
    os.makedirs(video_save_dir, exist_ok=True)
    filename = f"{frame_idx:04d}.png"
    save_path = os.path.join(video_save_dir, filename)
    if image.ndim == 3 and image.shape[2] == 3:
        image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_to_save = image
    cv2.imwrite(save_path, image_to_save, [cv2.IMWRITE_PNG_COMPRESSION, 0])



def _save_frame_png(image: np.ndarray, video_name: str, frame_name: str, save_dir: str) -> None:
    video_save_dir = os.path.join(save_dir, video_name)
    os.makedirs(video_save_dir, exist_ok=True)
    frame_stem = os.path.splitext(frame_name)[0]
    save_path = os.path.join(video_save_dir, f"{frame_stem}.png")
    image_to_save = np.asarray(image)
    if image_to_save.ndim == 3 and image_to_save.shape[2] == 3:
        image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image_to_save, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def _save_mask_png(mask: np.ndarray, video_name: str, frame_name: str, save_dir: str) -> None:
    mask_np = np.asarray(mask)
    if mask_np.ndim > 2:
        mask_np = np.squeeze(mask_np)
    if mask_np.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {tuple(mask_np.shape)}")
    mask_u8 = (mask_np > 0).astype(np.uint8) * 255
    _save_frame_png(mask_u8, video_name, frame_name, save_dir)


def _save_overlay_png(
    image: np.ndarray,
    mask: np.ndarray,
    video_name: str,
    frame_name: str,
    save_dir: str,
    overlay_color: Tuple[int, int, int],
    alpha: float = 0.45,
) -> None:
    image_rgb = np.asarray(image, dtype=np.uint8)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image for overlay save, got shape {tuple(image_rgb.shape)}")

    mask_arr = np.asarray(mask)
    if mask_arr.ndim > 2:
        mask_arr = np.squeeze(mask_arr)
    if mask_arr.ndim != 2:
        raise ValueError(f"Expected 2D mask for overlay save, got shape {tuple(mask_arr.shape)}")
    mask_bool = np.asarray(mask_arr, dtype=np.uint8) > 0

    overlay = image_rgb.astype(np.float32)
    if np.any(mask_bool):
        color = np.asarray(overlay_color, dtype=np.float32).reshape(1, 1, 3)
        overlay = np.where(mask_bool[..., None], overlay * (1.0 - alpha) + color * alpha, overlay)

    _save_frame_png(overlay.clip(0.0, 255.0).astype(np.uint8), video_name, frame_name, save_dir)


def _read_rgb_image(image_path: str) -> np.ndarray:
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read RGB image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _normalize_point_to_xy(point_arr: np.ndarray, width: int, height: int) -> np.ndarray:
    points = np.asarray(point_arr, dtype=np.float32).reshape(-1, 2)
    norm_points = points.copy()
    if norm_points.size == 0:
        raise ValueError('Point prompt is empty')
    if not np.all(np.isfinite(norm_points)):
        raise ValueError(f"Point prompt contains non-finite values: {norm_points.tolist()}")
    if np.max(np.abs(norm_points)) > 1.0:
        norm_points[:, 0] = norm_points[:, 0] / max(float(width), 1.0)
        norm_points[:, 1] = norm_points[:, 1] / max(float(height), 1.0)
    norm_points[:, 0] = np.clip(norm_points[:, 0], 1e-6, 1.0 - 1e-6)
    norm_points[:, 1] = np.clip(norm_points[:, 1], 1e-6, 1.0 - 1e-6)
    return norm_points.astype(np.float32)


def _normalize_box_to_xyxy(box_arr: np.ndarray, width: int, height: int) -> np.ndarray:
    box = np.asarray(box_arr, dtype=np.float32).reshape(-1)
    if box.shape[0] != 4:
        raise ValueError(f"Expected box with 4 elements, got {tuple(box.shape)}")
    if not np.all(np.isfinite(box)):
        raise ValueError(f"Box prompt contains non-finite values: {box.tolist()}")

    if np.max(np.abs(box)) > 1.0:
        box = np.array(
            [
                box[0] / max(float(width), 1.0),
                box[1] / max(float(height), 1.0),
                box[2] / max(float(width), 1.0),
                box[3] / max(float(height), 1.0),
            ],
            dtype=np.float32,
        )

    x0, y0, x1, y1 = [float(v) for v in box]
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    x0 = float(np.clip(x0, 1e-6, 1.0 - 1e-6))
    y0 = float(np.clip(y0, 1e-6, 1.0 - 1e-6))
    x1 = float(np.clip(max(x1, x0 + 1e-6), 1e-6, 1.0 - 1e-6))
    y1 = float(np.clip(max(y1, y0 + 1e-6), 1e-6, 1.0 - 1e-6))
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def _build_point_prompt_from_pixels(point_arr: np.ndarray, width: int, height: int, device: torch.device) -> Prompt:
    norm_points = _normalize_point_to_xy(point_arr, width, height)
    point_embeddings = torch.tensor([norm_points.tolist()], device=device, dtype=torch.float32)
    point_mask = torch.zeros(1, norm_points.shape[0], device=device, dtype=torch.bool)
    point_labels = torch.ones(1, norm_points.shape[0], device=device, dtype=torch.long)
    return Prompt(point_embeddings=point_embeddings, point_mask=point_mask, point_labels=point_labels)


def _build_box_prompt_from_pixels(box_arr: np.ndarray, width: int, height: int, device: torch.device) -> Prompt:
    x0, y0, x1, y1 = _normalize_box_to_xyxy(box_arr, width, height).tolist()
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    bw = x1 - x0
    bh = y1 - y0
    box_embeddings = torch.tensor([[[cx, cy, bw, bh]]], device=device, dtype=torch.float32)
    box_mask = torch.zeros(1, 1, device=device, dtype=torch.bool)
    box_labels = torch.ones(1, 1, device=device, dtype=torch.long)
    return Prompt(box_embeddings=box_embeddings, box_mask=box_mask, box_labels=box_labels)


def _compute_geom_prompt_score(
    image_rgb: np.ndarray,
    prompt_value: Any,
    prompt_mode: str,
    score_model,
) -> Optional[float]:
    if score_model is None:
        return None

    device = next(score_model.parameters()).device
    height, width = image_rgb.shape[:2]
    if prompt_mode == "pt":
        prompt = _build_point_prompt_from_pixels(prompt_value, width, height, device)
    elif prompt_mode == "bx":
        prompt = _build_box_prompt_from_pixels(prompt_value, width, height, device)
    else:
        return None

    image_tensor = torch.from_numpy(np.asarray(image_rgb, dtype=np.float32) / 255.0).permute(2, 0, 1).to(device)
    with torch.inference_mode():
        terms = forward_get_toplogit(
            score_model,
            image_tensor,
            "visual",
            prompt,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            encode_text=False,
        )
    if terms is None:
        return None
    score = terms.get("score")
    if score is None:
        return None
    score_value = float(score.detach().to(torch.float32).item())
    if not np.isfinite(score_value):
        return None
    return score_value


def _list_sorted_frame_names(video_dir: str) -> List[str]:
    frame_names = [
        name for name in os.listdir(video_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    try:
        frame_names.sort(key=lambda name: int(os.path.splitext(name)[0]))
    except ValueError:
        frame_names.sort()
    return frame_names


def _to_numpy(value: Any):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if torch.is_floating_point(tensor):
            tensor = tensor.to(torch.float32)
        return tensor.cpu().numpy()
    return np.asarray(value)


def _extract_frame_array(arr_all: Any, frame_pos: int, num_frames: Optional[int] = None):
    if arr_all is None:
        return None

    if isinstance(arr_all, (list, tuple)):
        if frame_pos < 0 or frame_pos >= len(arr_all):
            return None
        return _to_numpy(arr_all[frame_pos])

    array_value = _to_numpy(arr_all)
    if array_value is None:
        return None

    if num_frames is not None and array_value.ndim >= 1:
        if array_value.shape[0] == num_frames:
            return array_value[frame_pos] if 0 <= frame_pos < array_value.shape[0] else None
        if array_value.ndim >= 2 and array_value.shape[1] == num_frames:
            return array_value[:, frame_pos, ...] if 0 <= frame_pos < array_value.shape[1] else None

    return array_value


def _extract_text_scores(outputs: Dict[str, Any], frame_pos: int, num_frames: Optional[int] = None):
    scores = _extract_frame_array(outputs.get("out_probs"), frame_pos, num_frames)
    if scores is not None:
        return np.asarray(scores, dtype=np.float32).reshape(-1)

    pred_logits = _extract_frame_array(outputs.get("pred_logits"), frame_pos, num_frames)
    if pred_logits is None:
        return None

    pred_scores = torch.as_tensor(pred_logits, dtype=torch.float32).sigmoid().reshape(-1)
    presence_logits = _extract_frame_array(outputs.get("presence_logit_dec"), frame_pos, num_frames)
    if presence_logits is None:
        return pred_scores.cpu().numpy()

    presence_scores = torch.as_tensor(presence_logits, dtype=torch.float32).sigmoid().reshape(-1)
    if presence_scores.numel() == 1 and pred_scores.numel() > 1:
        presence_scores = presence_scores.expand(pred_scores.numel())
    if presence_scores.numel() != pred_scores.numel():
        return pred_scores.cpu().numpy()
    return (pred_scores * presence_scores).cpu().numpy()


def _extract_text_frame_outputs(outputs: Any, frame_pos: int, num_frames: Optional[int] = None):
    if outputs is None or not isinstance(outputs, dict):
        return None, None

    masks = _extract_frame_array(outputs.get("out_binary_masks"), frame_pos, num_frames)
    scores = _extract_text_scores(outputs, frame_pos, num_frames)

    if masks is not None:
        masks = np.asarray(masks)
        if masks.ndim == 2:
            masks = masks[None, ...]
        elif masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        elif masks.ndim != 3:
            masks = None
        elif masks.dtype != np.bool_:
            masks = masks > 0

    return masks, scores


def _normalize_video_masks(video_masks: Any) -> Optional[np.ndarray]:
    masks = _to_numpy(video_masks)
    if masks is None:
        return None

    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    elif masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    else:
        masks = np.squeeze(masks)
        if masks.ndim == 2:
            masks = masks[None, ...]
        elif masks.ndim != 3:
            return None

    return masks > 0


def _zero_mask_from_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image for zero-mask fallback: {image_path}")
    return np.zeros_like(image, dtype=np.uint8)


def _save_zero_mask_exports(
    video_dir: str,
    video_name: str,
    frame_names: List[str],
    mask_save_dir: str,
    score_dict: Dict[Tuple[str, str], Optional[float]],
    overlay_save_dir: Optional[str] = None,
    overlay_color: Tuple[int, int, int] = (0, 255, 0),
) -> None:
    for frame_name in frame_names:
        image_path = os.path.join(video_dir, frame_name)
        image_rgb = _read_rgb_image(image_path)
        zero_mask = _zero_mask_from_image(image_path)
        _save_mask_png(zero_mask, video_name, frame_name, mask_save_dir)
        if overlay_save_dir is not None:
            _save_overlay_png(image_rgb, zero_mask, video_name, frame_name, overlay_save_dir, overlay_color)
        score_dict[(video_name, frame_name)] = None


def _safe_close_text_session(video_predictor, session_id: Optional[str]) -> None:
    if session_id is None:
        return
    try:
        video_predictor.handle_request({"type": "close_session", "session_id": session_id})
    except Exception as exc:
        print(f"Warning: failed to close session {session_id}: {exc}")


def _is_missing_conditioning_error(exc: BaseException) -> bool:
    return "No points are provided; please add points first" in str(exc)


def _run_text_video_session_with_retry(video_predictor, video_dir: str, text_prompt: str, video_name: str):
    last_exc = None
    for attempt in range(2):
        session_id = None
        try:
            response = video_predictor.handle_request(
                request={"type": "start_session", "resource_path": video_dir}
            )
            session_id = response["session_id"]
            video_predictor.handle_request(
                request={
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": text_prompt,
                }
            )
            outputs_per_frame_raw = {}
            for response in video_predictor.handle_stream_request(
                request={"type": "propagate_in_video", "session_id": session_id}
            ):
                outputs_per_frame_raw[response["frame_index"]] = response["outputs"]
            return session_id, outputs_per_frame_raw
        except RuntimeError as exc:
            _safe_close_text_session(video_predictor, session_id)
            if not _is_missing_conditioning_error(exc):
                raise
            last_exc = exc
            print(
                f"[session-retry] {video_name}: invalid conditioning state on attempt {attempt + 1}/2: {exc}"
            )
            torch.cuda.empty_cache()
        except Exception:
            _safe_close_text_session(video_predictor, session_id)
            raise

    raise last_exc


def _select_text_mask_and_score(outputs_this_frame: Any, frame_pos: int, num_frames: int):
    masks, scores = _extract_text_frame_outputs(outputs_this_frame, frame_pos=0, num_frames=None)
    if masks is None and isinstance(outputs_this_frame, dict):
        masks, scores = _extract_text_frame_outputs(outputs_this_frame, frame_pos=frame_pos, num_frames=num_frames)

    if masks is None or masks.shape[0] == 0:
        return None, None

    best_idx = 0
    best_score = None
    if scores is not None and len(scores) == masks.shape[0]:
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

    return masks[best_idx].astype(bool), best_score


def _export_text_masks_and_scores(
    video_root_dir: str,
    mask_save_dir: str,
    start_P_dict: Dict[str, Any],
    video_predictor,
    overlay_save_dir: Optional[str] = None,
    overlay_color: Tuple[int, int, int] = (64, 224, 96),
) -> Dict[Tuple[str, str], Optional[float]]:
    frame_scores: Dict[Tuple[str, str], Optional[float]] = {}

    video_names = [
        name for name in os.listdir(video_root_dir)
        if os.path.isdir(os.path.join(video_root_dir, name))
    ]
    video_names.sort()

    for video_name in video_names:
        video_dir = os.path.join(video_root_dir, video_name)
        frame_names = _list_sorted_frame_names(video_dir)
        if not frame_names:
            continue

        prompt_value = start_P_dict.get(video_name)
        if prompt_value is None:
            print(f"[presence-export] missing prompt for {video_name}, saving zero masks")
            _save_zero_mask_exports(video_dir, video_name, frame_names, mask_save_dir, frame_scores, overlay_save_dir, overlay_color)
            continue

        text_prompt = str(prompt_value)
        session_id = None
        try:
            session_id, outputs_per_frame_raw = _run_text_video_session_with_retry(
                video_predictor, video_dir, text_prompt, video_name
            )

            for frame_pos, frame_name in enumerate(frame_names):
                image_path = os.path.join(video_dir, frame_name)
                image_rgb = _read_rgb_image(image_path)
                outputs_this_frame = outputs_per_frame_raw.get(frame_pos)
                top_mask, top_score = _select_text_mask_and_score(outputs_this_frame, frame_pos, len(frame_names))
                if top_mask is None:
                    top_mask = _zero_mask_from_image(image_path)
                _save_mask_png(top_mask, video_name, frame_name, mask_save_dir)
                if overlay_save_dir is not None:
                    _save_overlay_png(image_rgb, top_mask, video_name, frame_name, overlay_save_dir, overlay_color)
                frame_scores[(video_name, frame_name)] = top_score
        except RuntimeError as exc:
            if not _is_missing_conditioning_error(exc):
                raise
            print(f"[presence-export] {video_name}: {exc}; saving zero masks")
            _save_zero_mask_exports(video_dir, video_name, frame_names, mask_save_dir, frame_scores, overlay_save_dir, overlay_color)
        finally:
            _safe_close_text_session(video_predictor, session_id)
            torch.cuda.empty_cache()

    return frame_scores


def _select_geom_mask_and_score(video_res_masks: Any, obj_scores: Any):
    masks = _normalize_video_masks(video_res_masks)
    if masks is None or masks.shape[0] == 0:
        return None, None

    best_idx = 0
    best_score = None
    if obj_scores is not None:
        score_array = _to_numpy(obj_scores)
        if score_array is not None:
            score_tensor = torch.as_tensor(score_array, dtype=torch.float32)
            score_values = score_tensor.sigmoid().reshape(-1).cpu().numpy()
            if score_values.size == 1 and masks.shape[0] > 1:
                score_values = np.full(masks.shape[0], float(score_values[0]), dtype=np.float32)
            if score_values.size == masks.shape[0]:
                best_idx = int(np.argmax(score_values))
                best_score = float(score_values[best_idx])

    return masks[best_idx].astype(bool), best_score


def _export_geom_masks_and_scores(
    video_root_dir: str,
    mask_save_dir: str,
    start_P_dict: Dict[str, Any],
    predictor,
    score_model,
    prompt_mode: str,
    overlay_save_dir: Optional[str] = None,
    overlay_color: Tuple[int, int, int] = (64, 224, 96),
) -> Dict[Tuple[str, str], Optional[float]]:
    frame_scores: Dict[Tuple[str, str], Optional[float]] = {}

    video_names = [
        name for name in os.listdir(video_root_dir)
        if os.path.isdir(os.path.join(video_root_dir, name))
    ]
    video_names.sort()

    for video_name in video_names:
        video_dir = os.path.join(video_root_dir, video_name)
        frame_names = _list_sorted_frame_names(video_dir)
        if not frame_names:
            continue

        prompt_value = start_P_dict.get(video_name)
        if prompt_value is None:
            print(f"[presence-export] missing prompt for {video_name}, saving zero masks")
            _save_zero_mask_exports(video_dir, video_name, frame_names, mask_save_dir, frame_scores, overlay_save_dir, overlay_color)
            continue
        if not np.all(np.isfinite(np.asarray(prompt_value, dtype=np.float32))):
            print(f"[presence-export] invalid prompt for {video_name}: {prompt_value}; saving zero masks")
            _save_zero_mask_exports(video_dir, video_name, frame_names, mask_save_dir, frame_scores, overlay_save_dir, overlay_color)
            continue

        inference_state = predictor.init_state(video_path=video_dir)
        predictor.clear_all_points_in_video(inference_state)

        if prompt_mode == "pt":
            points = np.asarray(prompt_value, dtype=np.float32)
            labels = np.ones((points.reshape(-1, 2).shape[0],), np.int32)
            rel_points = [[x / 1008.0, y / 1008.0] for x, y in points]
            points_tensor = torch.tensor(rel_points, dtype=torch.float32)
            points_labels_tensor = torch.tensor(labels, dtype=torch.int32)
            predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=points_tensor,
                labels=points_labels_tensor,
                clear_old_points=False,
            )
        else:
            box = np.asarray(prompt_value, dtype=np.float32).reshape(-1)
            xmin, ymin, xmax, ymax = box
            rel_box = np.array([[xmin / 1008.0, ymin / 1008.0, xmax / 1008.0, ymax / 1008.0]], dtype=np.float32)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                box=rel_box,
            )

        frame_outputs: Dict[int, Tuple[Any, Any]] = {}
        max_frame_num_to_track = max(1, len(frame_names))
        for frame_pos, _, _, video_res_masks, obj_scores in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=False,
            propagate_preflight=True,
        ):
            frame_outputs[frame_pos] = (video_res_masks, obj_scores)

        for frame_pos, frame_name in enumerate(frame_names):
            image_path = os.path.join(video_dir, frame_name)
            image_rgb = _read_rgb_image(image_path)
            top_mask = None
            if frame_pos in frame_outputs:
                top_mask, _ = _select_geom_mask_and_score(*frame_outputs[frame_pos])
            if top_mask is None:
                top_mask = _zero_mask_from_image(image_path)
            top_score = _compute_geom_prompt_score(image_rgb, prompt_value, prompt_mode, score_model)
            _save_mask_png(top_mask, video_name, frame_name, mask_save_dir)
            if overlay_save_dir is not None:
                _save_overlay_png(image_rgb, top_mask, video_name, frame_name, overlay_save_dir, overlay_color)
            frame_scores[(video_name, frame_name)] = top_score

        torch.cuda.empty_cache()

    return frame_scores


def _frame_sort_key(item: Tuple[str, str]):
    video_name, frame_name = item
    frame_stem = os.path.splitext(frame_name)[0]
    try:
        return video_name, 0, int(frame_stem)
    except ValueError:
        return video_name, 1, frame_stem


def _format_optional_score(score: Optional[float]) -> str:
    if score is None or not np.isfinite(float(score)):
        return "NA"
    return f"{score:.6f}"


def _mean_optional_scores(score_dict: Dict[Tuple[str, str], Optional[float]]) -> Tuple[Optional[float], int]:
    values = [
        float(value)
        for value in score_dict.values()
        if value is not None and np.isfinite(float(value))
    ]
    if not values:
        return None, 0
    return float(np.mean(values)), len(values)


def _write_frame_presence_scores(
    csv_path: str,
    clean_scores: Dict[Tuple[str, str], Optional[float]],
    adv_scores: Dict[Tuple[str, str], Optional[float]],
) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["video_name", "frame_name", "frame_idx", "clean_presence_score", "adv_presence_score"]
    frame_keys = sorted(set(clean_scores) | set(adv_scores), key=_frame_sort_key)
    with open(csv_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for video_name, frame_name in frame_keys:
            writer.writerow(
                {
                    "video_name": video_name,
                    "frame_name": frame_name,
                    "frame_idx": os.path.splitext(frame_name)[0],
                    "clean_presence_score": _format_optional_score(clean_scores.get((video_name, frame_name))),
                    "adv_presence_score": _format_optional_score(adv_scores.get((video_name, frame_name))),
                }
            )


def _print_frame_presence_scores(
    clean_scores: Dict[Tuple[str, str], Optional[float]],
    adv_scores: Dict[Tuple[str, str], Optional[float]],
) -> None:
    frame_keys = sorted(set(clean_scores) | set(adv_scores), key=_frame_sort_key)
    for video_name, frame_name in frame_keys:
        clean_score = _format_optional_score(clean_scores.get((video_name, frame_name)))
        adv_score = _format_optional_score(adv_scores.get((video_name, frame_name)))
        print(f"[presence] {video_name}/{frame_name}: clean={clean_score}, adv={adv_score}")
def _normalize_prompt_value(prompt_value: Any, prompt_mode: str, context: str):
    if prompt_mode == "text":
        prompt_text = parse_text_prompt_value(prompt_value)
        if not prompt_text:
            raise ValueError(f"[{context}] text prompt is empty after normalization")
        return prompt_text

    arr = np.asarray(prompt_value, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"[{context}] prompt contains non-finite values: {arr.tolist()}")
    if prompt_mode == "pt":
        if arr.ndim == 1:
            if arr.shape[0] != 2:
                raise ValueError(
                    f"[{context}] test_prompts=pt expects shape (2,) or (N,2), got {tuple(arr.shape)}"
                )
            return arr.reshape(1, 2)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr
        raise ValueError(f"[{context}] test_prompts=pt expects shape (2,) or (N,2), got {tuple(arr.shape)}")

    flat = arr.reshape(-1)
    if flat.shape[0] != 4:
        raise ValueError(f"[{context}] test_prompts=bx expects shape (4,), got {tuple(arr.shape)}")
    return flat


def _write_prob_stats(stats_dir: str, prob_below_clean: int, prob_below_adv: int, count_clean: int, count_adv: int) -> None:
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, "predprob_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("out_probs stats\n")
        f.write(
            f"out_probs<0.5 clean: {prob_below_clean}/{count_clean}, "
            f"adv: {prob_below_adv}/{count_adv}\n"
        )
    print(f"Saved stats to {stats_path}")


def _build_eval_components(prompt_mode: str):
    if prompt_mode == "text":
        return build_sam3_video_predictor(), None

    sam3_model = build_sam3_video_model()
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    score_model = sam3_model.detector
    return predictor, score_model


def main():
    args = parse_args()
    _validate_and_normalize_args(args)
    run_tag = _resolve_run_tag(args)
    seed_everything(seed=args.seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # load delta
    ckpt = torch.load(args.delta_path, map_location=device)
    delta = ckpt.get("delta", None)
    if delta is None:
        raise ValueError(f"delta not found in {args.delta_path}")
    delta = _normalize_delta_to_chw(delta).to(device).detach()  # keep delta on device but out of autograd
    delta.requires_grad_(False)
    delta_size = tuple(delta.shape[-2:])

    # predictor for evaluation
    predictor, prompt_score_model = _build_eval_components(args.test_prompts)
    if args.test_prompts == "pt" and args.enable_multi_point_eval:
        predictor = _enable_multi_point_label_broadcast(predictor)
        print(f"[multi-point] enabled foreground point sampling with count={args.multi_point_num}")

    # dataset / loader
    custom_dataset = choose_dataset(args)
    video_to_indices = get_video_to_indices(custom_dataset)

    # paths
    base_root = str(DEFAULT_OUTPUT_ROOT)
    video_dir_prefix = "finalvideo-"
    result_path = os.path.join(
        base_root,
        f"{video_dir_prefix}adv_{args.test_prompts}_{args.train_dataset}_{args.checkpoints}_{args.seed}_{run_tag}",
    )
    result_path_clean = os.path.join(
        base_root,
        f"{video_dir_prefix}clean_{args.test_prompts}_{args.train_dataset}_{args.checkpoints}_{args.seed}_{run_tag}",
    )
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(result_path_clean, exist_ok=True)
    mask_path_clean = os.path.join(
        base_root,
        f"{video_dir_prefix}cleanmask_{args.test_prompts}_{args.train_dataset}_{args.checkpoints}_{args.seed}_{run_tag}",
    )
    mask_path_adv = os.path.join(
        base_root,
        f"{video_dir_prefix}advmask_{args.test_prompts}_{args.train_dataset}_{args.checkpoints}_{args.seed}_{run_tag}",
    )
    overlay_path_clean = os.path.join(
        base_root,
        f"{video_dir_prefix}cleanmaskoverlay_{args.test_prompts}_{args.train_dataset}_{args.checkpoints}_{args.seed}_{run_tag}",
    )
    overlay_path_adv = os.path.join(
        base_root,
        f"{video_dir_prefix}advmaskoverlay_{args.test_prompts}_{args.train_dataset}_{args.checkpoints}_{args.seed}_{run_tag}",
    )

    log_save_path = os.path.join("result", "uapdelta", "log")
    os.makedirs(log_save_path, exist_ok=True)
    now_time = make_print_to_file(path=log_save_path)

    printed_videos = set()
    mask_gt_dict: Dict[Tuple[str, int], np.ndarray] = {}
    frame_counter: Dict[str, int] = {}
    start_P_dict: Dict[str, Any] = {}

    # iterate videos
    with torch.inference_mode():
        for video_name, indices in video_to_indices.items():
            print(f"\n{'=' * 40} Processing video: {video_name} {'=' * 40}")
            video_subset = Subset(custom_dataset, indices)
            video_loader = DataLoader(video_subset, batch_size=1, collate_fn=collate_fn, shuffle=False, num_workers=0)

            start_frame_idx = custom_dataset.get_start_frame_idx(video_name)

            for images, P_list, img_ids, gt, point in video_loader:
                img_ID, img, mask_gt, P_gt = img_ids[0], images[0], gt[0], P_list[0]

                # frame index
                if args.train_dataset == "segtrack":
                    frame_idx = get_frame_index_seg(img_ID)
                else:
                    frame_idx = get_frame_index(img_ID)

                print(f"Processing frame {frame_idx} from video {video_name}")

                if video_name not in printed_videos:
                    if start_frame_idx is not None:
                        print(f"Start frame index for video {video_name}: {start_frame_idx}")
                    else:
                        print(f"Could not find start frame index for video {video_name}")
                    printed_videos.add(video_name)

                # per-video counters
                if video_name not in frame_counter:
                    frame_counter[video_name] = 0
                indexed_frame_idx = frame_counter[video_name]
                frame_counter[video_name] += 1

                mask_gt_dict[(video_name, indexed_frame_idx)] = mask_gt

                # record dataset prompt once per video using the annotated start frame
                if frame_idx == start_frame_idx and video_name not in start_P_dict:
                    prompt_source = point[0]
                    prompt_source_desc = "dataset prompt"
                    if args.test_prompts == "pt" and args.enable_multi_point_eval:
                        try:
                            prompt_source = _sample_foreground_points_from_mask(
                                mask_gt,
                                args.multi_point_num,
                                f"{video_name}:{frame_idx}",
                            )
                            prompt_source_desc = f"mask_gt foreground points x{prompt_source.shape[0]}"
                        except ValueError as exc:
                            print(f"[warn] multi-point sampling failed for {video_name}, falling back to dataset prompt: {exc}")
                    try:
                        prompt_value = _normalize_prompt_value(prompt_source, args.test_prompts, f"{video_name}:{frame_idx}")
                    except ValueError as exc:
                        print(f"[warn] skipping invalid start prompt for {video_name}: {exc}")
                    else:
                        start_P_dict[video_name] = prompt_value
                        print(f"Recorded start prompt for {video_name} ({prompt_source_desc}): {prompt_value}")

                # tensors (CPU is enough; avoid needless autograd graphs)
                img_np = np.asarray(img).astype(np.float32) / 255.0
                img_t = torch.from_numpy(img_np).permute(2, 0, 1)
                img_t = _ensure_size(img_t, delta_size)

                # save clean (lossless PNG)
                im = _tensor_to_uint8_hwc(img_t)
                _save_image_png(im, video_name, frame_idx, result_path_clean)

                # adv
                adv_t = torch.clamp(img_t + delta.cpu(), 0.0, 1.0)

                adv_image = _tensor_to_uint8_hwc(adv_t)
                _save_image_png(adv_image, video_name, frame_idx, result_path)

                # cleanup
                del img_t, adv_t

            torch.cuda.empty_cache()

    # evaluation
    output_dir_adv = os.path.join(
        base_root,
        f"{video_dir_prefix}save_image_adv_{args.test_prompts}_{args.train_dataset}_{args.checkpoints}_{args.seed}_{run_tag}",
    )
    output_dir_clean = os.path.join(
        base_root,
        f"{video_dir_prefix}save_image_clean_{args.test_prompts}_{args.train_dataset}_{args.checkpoints}_{args.seed}_{run_tag}",
    )

    prob_below_clean = None
    prob_below_adv = None

    if args.test_prompts == "text":
        (
            miou_all_clean,
            iou_count_all,
            skipped_frames_all,
            frame_iou_dict_clean,
            frame_prob_dict_clean,
            prob_below_clean,
        ) = process_videos_text(
            result_path_clean,
            output_dir_clean,
            mask_gt_dict,
            start_P_dict,
            predictor,
            category="clean",
            args=args,
        )

        (
            miou_all_adv,
            iou_count_all_adv,
            _,
            frame_iou_dict_adv,
            frame_prob_dict_adv,
            prob_below_adv,
        ) = process_videos_text(
            result_path,
            output_dir_adv,
            mask_gt_dict,
            start_P_dict,
            predictor,
            category="adversarial",
            skipped_frames=skipped_frames_all,
            args=args,
        )
    else:
        miou_all_clean, iou_count_all, skipped_frames_all = process_videos(
            result_path_clean,
            output_dir_clean,
            mask_gt_dict,
            start_P_dict,
            predictor,
            category="clean",
            args=args,
        )
        miou_all_adv, iou_count_all_adv, _ = process_videos(
            result_path,
            output_dir_adv,
            mask_gt_dict,
            start_P_dict,
            predictor,
            category="adversarial",
            skipped_frames=skipped_frames_all,
            args=args,
        )

    print(f">> all: miou_clean: {miou_all_clean}, miou_adv: {miou_all_adv}, count_clean: {iou_count_all}, count_adv: {iou_count_all_adv}")

    stats_dir = os.path.join("result", "uapdelta")
    if prob_below_clean is not None and prob_below_adv is not None:
        high_conf_clean = max(iou_count_all - prob_below_clean, 0)
        high_conf_adv = max(iou_count_all_adv - prob_below_adv, 0)
        print(
            f"out_probs<0.5 — clean: {prob_below_clean}/{iou_count_all}, adv: {prob_below_adv}/{iou_count_all_adv}"
        )
        print(
            f"high-confidence counted frames (out_probs>=0.5) — clean: {high_conf_clean}/{iou_count_all}, adv: {high_conf_adv}/{iou_count_all_adv}"
        )
        print(
            "[stats] count_clean/count_adv include every counted frame with a predicted mask, even when out_probs<0.5. "
            "Use the high-confidence counts above if you want the stricter confidence-filtered subset."
        )
        _write_prob_stats(stats_dir, prob_below_clean, prob_below_adv, iou_count_all, iou_count_all_adv)

    print("Exporting frame-level masks, overlays, and presence scores...")
    if args.test_prompts == "text":
        frame_presence_scores_clean = _export_text_masks_and_scores(
            result_path_clean,
            mask_path_clean,
            start_P_dict,
            predictor,
            overlay_save_dir=overlay_path_clean,
            overlay_color=(64, 224, 96),
        )
        frame_presence_scores_adv = _export_text_masks_and_scores(
            result_path,
            mask_path_adv,
            start_P_dict,
            predictor,
            overlay_save_dir=overlay_path_adv,
            overlay_color=(255, 96, 64),
        )
    else:
        frame_presence_scores_clean = _export_geom_masks_and_scores(
            result_path_clean,
            mask_path_clean,
            start_P_dict,
            predictor,
            prompt_score_model,
            args.test_prompts,
            overlay_save_dir=overlay_path_clean,
            overlay_color=(64, 224, 96),
        )
        frame_presence_scores_adv = _export_geom_masks_and_scores(
            result_path,
            mask_path_adv,
            start_P_dict,
            predictor,
            prompt_score_model,
            args.test_prompts,
            overlay_save_dir=overlay_path_adv,
            overlay_color=(255, 96, 64),
        )
    clean_presence_mean, clean_presence_count = _mean_optional_scores(frame_presence_scores_clean)
    adv_presence_mean, adv_presence_count = _mean_optional_scores(frame_presence_scores_adv)
    print(
        f">> presence: clean_mean: {_format_optional_score(clean_presence_mean)}, "
        f"adv_mean: {_format_optional_score(adv_presence_mean)}, "
        f"count_clean: {clean_presence_count}, count_adv: {adv_presence_count}"
    )
    print(f"Saved clean masks to {mask_path_clean}")
    print(f"Saved adv masks to {mask_path_adv}")
    print(f"Saved clean mask overlays to {overlay_path_clean}")
    print(f"Saved adv mask overlays to {overlay_path_adv}")

    if args.save:
        final_log_save_path = os.path.join("result", "uapdelta")
        os.makedirs(final_log_save_path, exist_ok=True)
        final_result = [
            {
                "seed": args.seed,
                "now_time": now_time,
                "delta_path": args.delta_path,
                "train_dataset": args.train_dataset,
                "test_prompts": args.test_prompts,
                "miou_clean": f"{miou_all_clean * 100:.2f} %",
                "miou_adv": f"{miou_all_adv * 100:.2f} %",
                "presence_clean_mean": _format_optional_score(clean_presence_mean),
                "presence_adv_mean": _format_optional_score(adv_presence_mean),
                "presence_clean_count": clean_presence_count,
                "presence_adv_count": adv_presence_count,
                "count_clean": iou_count_all,
                "count_adv": iou_count_all_adv,
            }
        ]
        csv_path = os.path.join(final_log_save_path, "final_results.csv")

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=final_result[0].keys())
            writer.writeheader()
            writer.writerows(final_result)
        print(f"Saved summary to {csv_path}")


if __name__ == "__main__":
    main()
