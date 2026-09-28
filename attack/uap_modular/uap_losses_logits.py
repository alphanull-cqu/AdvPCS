"""Pred-logit, consistency and presence-logit losses for UAP attack."""

from typing import Any, Dict, List, Optional, Tuple
from contextlib import nullcontext

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from sam3_util import get_frame_index, get_frame_index_seg


def prepare_image(img: Image.Image, image_size: int, device: torch.device) -> Tuple[torch.Tensor, int, int]:
    img_np = np.asarray(img).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).to(device)
    orig_h, orig_w = img_t.shape[-2:]
    img_t = v2.functional.resize(img_t, size=(image_size, image_size))
    return img_t, orig_h, orig_w


def norm_by_mean_std(img_01: torch.Tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=img_01.device, dtype=img_01.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, device=img_01.device, dtype=img_01.dtype).view(3, 1, 1)
    return (img_01 - mean_t) / std_t


def build_empty_geometric_prompt(device: torch.device):
    from sam3.model.geometry_encoders import Prompt

    return Prompt(
        box_embeddings=torch.zeros(0, 1, 4, device=device),
        box_mask=torch.zeros(1, 0, device=device, dtype=torch.bool),
    )


def parse_text_prompt(prompt_obj: Any) -> str:
    if isinstance(prompt_obj, str):
        return prompt_obj
    if isinstance(prompt_obj, dict):
        for key in ("text", "prompt", "caption", "query"):
            if key in prompt_obj and isinstance(prompt_obj[key], str):
                return prompt_obj[key]
    if isinstance(prompt_obj, (list, tuple)) and len(prompt_obj) > 0 and isinstance(prompt_obj[0], str):
        return prompt_obj[0]
    return str(prompt_obj)


def extract_img_id_str(img_ids: Any) -> str:
    if isinstance(img_ids, (list, tuple)):
        return str(img_ids[0]) if len(img_ids) > 0 else "unknown"
    if isinstance(img_ids, str):
        return img_ids
    return "unknown"


def extract_video_name_and_frame_idx(img_id_str: str) -> Tuple[str, int]:
    video_name = img_id_str.split("/")[0] if isinstance(img_id_str, str) and len(img_id_str) > 0 else "unknown"
    frame_idx = 0
    try:
        frame_idx = int(get_frame_index(img_id_str))
    except Exception:
        try:
            frame_idx = int(get_frame_index_seg(img_id_str))
        except Exception:
            frame_idx = 0
    return video_name, frame_idx


def mask_to_numpy_bool(mask_data: Any):
    if mask_data is None:
        return None

    if torch.is_tensor(mask_data):
        mask_np = mask_data.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask_data)

    if mask_np.size == 0:
        return None

    if mask_np.ndim == 3:
        if mask_np.shape[0] <= 4 and mask_np.shape[-1] > 4:
            mask_np = mask_np[0]
        elif mask_np.shape[-1] <= 4:
            mask_np = mask_np[..., 0]
        else:
            mask_np = mask_np[0]
    elif mask_np.ndim > 3:
        mask_np = np.squeeze(mask_np)
        if mask_np.ndim > 2:
            mask_np = mask_np.reshape(mask_np.shape[-2], mask_np.shape[-1])

    if mask_np.ndim != 2:
        return None

    fg = mask_np > 0
    if not np.any(fg):
        return None
    return fg


def _pixel_to_norm_xy(x: float, y: float, width: int, height: int) -> Tuple[float, float]:
    denom_w = max(float(width), 1.0)
    denom_h = max(float(height), 1.0)
    px = float(np.clip((float(x) + 0.5) / denom_w, 1e-6, 1.0 - 1e-6))
    py = float(np.clip((float(y) + 0.5) / denom_h, 1e-6, 1.0 - 1e-6))
    return px, py


def _select_foreground_pixel(xs, ys, x_target: float, y_target: float) -> Tuple[int, int]:
    d2 = (xs.astype(np.float32) - float(x_target)) ** 2 + (ys.astype(np.float32) - float(y_target)) ** 2
    best_idx = int(np.argmin(d2))
    return int(xs[best_idx]), int(ys[best_idx])


def extract_point_box_from_mask(mask_data: Any):
    if mask_data is None:
        return None

    if torch.is_tensor(mask_data):
        mask_np = mask_data.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask_data)

    if mask_np.size == 0:
        return None

    if mask_np.ndim == 3:
        if mask_np.shape[0] <= 4 and mask_np.shape[-1] > 4:
            mask_np = mask_np[0]
        elif mask_np.shape[-1] <= 4:
            mask_np = mask_np[..., 0]
        else:
            mask_np = mask_np[0]
    elif mask_np.ndim > 3:
        mask_np = np.squeeze(mask_np)
        if mask_np.ndim > 2:
            mask_np = mask_np.reshape(mask_np.shape[-2], mask_np.shape[-1])

    if mask_np.ndim != 2:
        return None

    fg = mask_np > 0
    ys, xs = np.where(fg)
    if len(xs) == 0:
        return None

    h, w = fg.shape
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())

    denom_w = max(float(w), 1.0)
    denom_h = max(float(h), 1.0)

    cx = ((x0 + x1) * 0.5) / denom_w
    cy = ((y0 + y1) * 0.5) / denom_h
    bw = (x1 - x0 + 1.0) / denom_w
    bh = (y1 - y0 + 1.0) / denom_h

    px = float(xs.mean()) / denom_w
    py = float(ys.mean()) / denom_h

    cx = float(np.clip(cx, 1e-6, 1.0 - 1e-6))
    cy = float(np.clip(cy, 1e-6, 1.0 - 1e-6))
    bw = float(np.clip(bw, 1e-6, 1.0))
    bh = float(np.clip(bh, 1e-6, 1.0))
    px = float(np.clip(px, 1e-6, 1.0 - 1e-6))
    py = float(np.clip(py, 1e-6, 1.0 - 1e-6))
    return cx, cy, bw, bh, px, py


def build_point_prompt_from_mask(mask_data: Any, device: torch.device):
    from sam3.model.geometry_encoders import Prompt

    point_box = extract_point_box_from_mask(mask_data)
    if point_box is None:
        return None
    _, _, _, _, px, py = point_box

    point_embeddings = torch.tensor([[[px, py]]], device=device, dtype=torch.float32)
    point_mask = torch.zeros(1, 1, device=device, dtype=torch.bool)
    point_labels = torch.ones(1, 1, device=device, dtype=torch.long)
    return Prompt(point_embeddings=point_embeddings, point_mask=point_mask, point_labels=point_labels)


def build_point_prompt_from_xy_norm(point_xy: Tuple[float, float], device: torch.device):
    from sam3.model.geometry_encoders import Prompt

    px, py = point_xy
    point_embeddings = torch.tensor([[[px, py]]], device=device, dtype=torch.float32)
    point_mask = torch.zeros(1, 1, device=device, dtype=torch.bool)
    point_labels = torch.ones(1, 1, device=device, dtype=torch.long)
    return Prompt(point_embeddings=point_embeddings, point_mask=point_mask, point_labels=point_labels)


def build_box_prompt_from_mask(mask_data: Any, device: torch.device):
    from sam3.model.geometry_encoders import Prompt

    point_box = extract_point_box_from_mask(mask_data)
    if point_box is None:
        return None
    cx, cy, bw, bh, _, _ = point_box

    box_embeddings = torch.tensor([[[cx, cy, bw, bh]]], device=device, dtype=torch.float32)
    box_mask = torch.zeros(1, 1, device=device, dtype=torch.bool)
    box_labels = torch.ones(1, 1, device=device, dtype=torch.long)
    return Prompt(box_embeddings=box_embeddings, box_mask=box_mask, box_labels=box_labels)


def build_box_prompt_from_xyxy_norm(box_xyxy: Tuple[float, float, float, float], device: torch.device):
    from sam3.model.geometry_encoders import Prompt

    x0, y0, x1, y1 = box_xyxy
    x0 = float(np.clip(x0, 1e-6, 1.0 - 1e-6))
    y0 = float(np.clip(y0, 1e-6, 1.0 - 1e-6))
    x1 = float(np.clip(x1, x0 + 1e-6, 1.0 - 1e-6))
    y1 = float(np.clip(y1, y0 + 1e-6, 1.0 - 1e-6))
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    bw = x1 - x0
    bh = y1 - y0

    box_embeddings = torch.tensor([[[cx, cy, bw, bh]]], device=device, dtype=torch.float32)
    box_mask = torch.zeros(1, 1, device=device, dtype=torch.bool)
    box_labels = torch.ones(1, 1, device=device, dtype=torch.long)
    return Prompt(box_embeddings=box_embeddings, box_mask=box_mask, box_labels=box_labels)


def build_point_prompt_candidates_from_mask(mask_data: Any, device: torch.device) -> List[Dict[str, Any]]:
    fg = mask_to_numpy_bool(mask_data)
    if fg is None:
        return []

    ys, xs = np.where(fg)
    if len(xs) == 0:
        return []

    h, w = fg.shape
    x_values = xs.astype(np.float32)
    y_values = ys.astype(np.float32)
    if len(x_values) > 1:
        x_targets = np.quantile(x_values, [0.2, 0.8])
    else:
        x_targets = np.asarray([x_values[0]] * 2)
    y_target = float(np.median(y_values))

    default_entry = build_default_point_prompt_entry(mask_data, device)
    candidates: List[Dict[str, Any]] = []
    seen = set()
    for idx, (label, x_target) in ((1, ("left", x_targets[0])), (3, ("right", x_targets[-1]))):
        cand_x, cand_y = _select_foreground_pixel(xs, ys, x_target, y_target)
        px, py = _pixel_to_norm_xy(cand_x, cand_y, w, h)
        key = (round(px, 6), round(py, 6))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "prompt": build_point_prompt_from_xy_norm((px, py), device),
                "index": float(idx),
                "label": label,
                "point_xy": (px, py),
            }
        )

    if default_entry is not None:
        default_key = tuple(round(v, 6) for v in default_entry["point_xy"])
        if default_key not in seen:
            seen.add(default_key)
            candidates.append(
                {
                    "prompt": default_entry["prompt"],
                    "index": 2.0,
                    "label": "default",
                    "point_xy": default_entry["point_xy"],
                }
            )

    if len(candidates) == 0:
        if default_entry is not None:
            candidates.append(
                {
                    "prompt": default_entry["prompt"],
                    "index": 2.0,
                    "label": "default",
                    "point_xy": default_entry["point_xy"],
                }
            )
    candidates.sort(key=lambda item: float(item.get("index", 999.0)))
    return candidates


def build_box_prompt_candidates_from_mask(mask_data: Any, device: torch.device) -> List[Dict[str, Any]]:
    default_entry = build_default_box_prompt_entry(mask_data, device)
    if default_entry is None:
        return []

    x0, y0, x1, y1 = default_entry["box_xyxy"]
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    bw = x1 - x0
    bh = y1 - y0

    candidates: List[Dict[str, Any]] = [
        {
            "prompt": default_entry["prompt"],
            "index": 1.0,
            "label": "default",
            "box_xyxy": default_entry["box_xyxy"],
        }
    ]
    for idx, (label, expand_ratio) in enumerate((("medium", 0.15), ("large", 0.30)), start=2):
        half_w = 0.5 * bw * (1.0 + float(expand_ratio))
        half_h = 0.5 * bh * (1.0 + float(expand_ratio))
        cand_xyxy = (
            float(np.clip(cx - half_w, 1e-6, 1.0 - 1e-6)),
            float(np.clip(cy - half_h, 1e-6, 1.0 - 1e-6)),
            float(np.clip(cx + half_w, 1e-6, 1.0 - 1e-6)),
            float(np.clip(cy + half_h, 1e-6, 1.0 - 1e-6)),
        )
        candidates.append(
            {
                "prompt": build_box_prompt_from_xyxy_norm(cand_xyxy, device),
                "index": float(idx),
                "label": label,
                "box_xyxy": cand_xyxy,
            }
        )

    return candidates


def build_default_point_prompt_entry(mask_data: Any, device: torch.device) -> Optional[Dict[str, Any]]:
    prompt = build_point_prompt_from_mask(mask_data, device)
    point_box = extract_point_box_from_mask(mask_data)
    if prompt is None or point_box is None:
        return None
    _, _, _, _, px, py = point_box
    return {
        "prompt": prompt,
        "index": 0.0,
        "label": "default",
        "point_xy": (float(px), float(py)),
    }


def build_default_box_prompt_entry(mask_data: Any, device: torch.device) -> Optional[Dict[str, Any]]:
    prompt = build_box_prompt_from_mask(mask_data, device)
    point_box = extract_point_box_from_mask(mask_data)
    if prompt is None or point_box is None:
        return None
    cx, cy, bw, bh, _, _ = point_box
    return {
        "prompt": prompt,
        "index": 0.0,
        "label": "default",
        "box_xyxy": (cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh),
    }


def build_text_prompt_candidate_entries(prompt_obj: Any, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
    try:
        from text_prompt_variants import extract_text_prompt_candidates
    except Exception:
        extract_text_prompt_candidates = None

    dataset_hint = dataset_name
    if isinstance(prompt_obj, dict):
        dataset_hint = prompt_obj.get("dataset") or dataset_hint

    if extract_text_prompt_candidates is not None:
        texts = extract_text_prompt_candidates(prompt_obj, dataset_name=dataset_hint)
    else:
        texts = [parse_text_prompt(prompt_obj)]

    candidates: List[Dict[str, Any]] = []
    seen = set()
    for text in texts:
        text_norm = str(text).strip()
        if len(text_norm) == 0 or text_norm in seen:
            continue
        seen.add(text_norm)
        next_idx = len(candidates) + 1
        candidates.append(
            {
                "text": text_norm,
                "index": float(next_idx),
                "label": "base" if len(candidates) == 0 else f"syn{len(candidates)}",
            }
        )
    return candidates


def compute_target_pred(
    attack_mode: str,
    target_pred_value: Optional[float],
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    if target_pred_value is None:
        target_pred_value = 0.01 if attack_mode == "suppress" else 0.99
    target_pred_value = max(0.01, min(0.99, float(target_pred_value)))
    target_pred_logit = torch.tensor(
        np.log(target_pred_value) - np.log(1 - target_pred_value),
        device=device,
        dtype=torch.float32,
    )
    return target_pred_logit, target_pred_value


def compute_prompt_selection_score(
    top_logit: Optional[torch.Tensor],
    presence_logit: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if top_logit is None:
        return None
    pred_prob = top_logit.sigmoid()
    if presence_logit is None:
        return pred_prob
    return pred_prob * presence_logit.sigmoid()


def _flatten_pred_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 3:
        return logits[0, :, 0]
    if logits.dim() == 2:
        if logits.shape[-1] == 1:
            return logits[:, 0]
        if logits.shape[0] == 1:
            return logits[0, :]
    return logits.reshape(-1)


def _flatten_presence_logits(
    presence_logits: Optional[torch.Tensor],
    expected_numel: int,
) -> Optional[torch.Tensor]:
    if presence_logits is None or presence_logits.numel() == 0:
        return None

    presence_flat = presence_logits.reshape(-1)
    if presence_flat.numel() == expected_numel:
        return presence_flat
    if presence_flat.numel() == 1:
        return presence_flat.expand(expected_numel)

    squeezed_flat = presence_logits.squeeze().reshape(-1)
    if squeezed_flat.numel() == expected_numel:
        return squeezed_flat
    if squeezed_flat.numel() == 1:
        return squeezed_flat.expand(expected_numel)

    return None


def forward_get_toplogit(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    text_prompt: str,
    geometric_prompt,
    mean,
    std,
    encode_text: bool = True,
) -> Optional[Dict[str, Optional[torch.Tensor]]]:
    try:
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        images = norm_by_mean_std(image_tensor, mean=mean, std=std)
        device = images.device
        prompt_text = str(text_prompt if encode_text else "visual")
        prompt_struct = geometric_prompt if geometric_prompt is not None else build_empty_geometric_prompt(device)
        autocast_ctx = torch.autocast(device_type="cuda", enabled=False) if images.is_cuda else nullcontext()

        with autocast_ctx:
            images = images.to(dtype=torch.float32)
            backbone_out = model.backbone.forward_image(images)
            text_outputs = model.backbone.forward_text([prompt_text], device=device)
            backbone_out.update(text_outputs)

            from sam3.model.data_misc import FindStage

            find_input = FindStage(
                img_ids=torch.tensor([0], device=device, dtype=torch.long),
                text_ids=torch.tensor([0], device=device, dtype=torch.long),
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )

            outputs = model.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_input,
                geometric_prompt=prompt_struct,
                find_target=None,
            )

        if "pred_logits" not in outputs:
            print(f"pred_logits missing. Keys: {list(outputs.keys())}")
            return None

        logits_last = _flatten_pred_logits(outputs["pred_logits"][-1])

        if logits_last.numel() == 0:
            print("pred_logits empty")
            return None

        pred_probs = logits_last.sigmoid()
        presence_top = None
        proposal_scores = pred_probs
        presence_logits = _flatten_presence_logits(outputs.get("presence_logit_dec"), logits_last.numel())
        if presence_logits is not None:
            proposal_scores = pred_probs * presence_logits.sigmoid()

        best_idx = proposal_scores.argmax()
        top_logit = logits_last[best_idx]
        if "presence_logit_dec" in outputs and outputs["presence_logit_dec"] is not None:
            if presence_logits is not None:
                presence_top = presence_logits[best_idx]

        return {"top_logit": top_logit, "presence_logit": presence_top, "score": proposal_scores[best_idx]}

    except Exception as exc:
        import traceback

        print(f"Error in forward pred-only: {exc}")
        traceback.print_exc()
        return None


def select_best_geometric_prompt_candidate(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    mean,
    std,
    candidates: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if len(candidates) == 0:
        return None, None

    if len(candidates) == 1:
        only_candidate = candidates[0]
        final_terms = forward_get_toplogit(
            model,
            image_tensor,
            "visual",
            only_candidate["prompt"],
            mean,
            std,
            encode_text=False,
        )
        if final_terms is not None:
            final_terms["selected_candidate_index"] = float(only_candidate.get("index", -1.0))
            final_terms["selected_candidate_label"] = str(only_candidate.get("label", ""))
        return only_candidate, final_terms

    best_candidate = None
    best_score = -1.0
    detached_image = image_tensor.detach()
    for candidate in candidates:
        probe_terms = forward_get_toplogit(
            model,
            detached_image,
            "visual",
            candidate["prompt"],
            mean,
            std,
            encode_text=False,
        )
        if probe_terms is None:
            continue
        selection_score = compute_prompt_selection_score(
            probe_terms.get("top_logit"),
            probe_terms.get("presence_logit"),
        )
        if selection_score is None:
            continue
        probe_score = float(selection_score.detach().item())
        if best_candidate is None or probe_score > best_score:
            best_candidate = candidate
            best_score = probe_score

    if best_candidate is None:
        best_candidate = candidates[0]

    final_terms = forward_get_toplogit(
        model,
        image_tensor,
        "visual",
        best_candidate["prompt"],
        mean,
        std,
        encode_text=False,
    )
    if final_terms is not None:
        final_terms["selected_candidate_index"] = float(best_candidate.get("index", -1.0))
        final_terms["selected_candidate_label"] = str(best_candidate.get("label", ""))
        final_score = compute_prompt_selection_score(
            final_terms.get("top_logit"),
            final_terms.get("presence_logit"),
        )
        final_terms["selection_score"] = float(
            final_score.detach().item() if final_score is not None else best_score
        )
    return best_candidate, final_terms


def select_best_text_prompt_candidate(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    mean,
    std,
    candidates: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if len(candidates) == 0:
        return None, None

    geometric_prompt = build_empty_geometric_prompt(image_tensor.device)
    if len(candidates) == 1:
        only_candidate = candidates[0]
        final_terms = forward_get_toplogit(
            model,
            image_tensor,
            only_candidate["text"],
            geometric_prompt,
            mean,
            std,
            encode_text=True,
        )
        if final_terms is not None:
            final_terms["selected_candidate_index"] = float(only_candidate.get("index", -1.0))
            final_terms["selected_candidate_label"] = str(only_candidate.get("label", ""))
            final_terms["selected_candidate_text"] = str(only_candidate.get("text", ""))
        return only_candidate, final_terms

    best_candidate = None
    best_score = -1.0
    detached_image = image_tensor.detach()
    for candidate in candidates:
        probe_terms = forward_get_toplogit(
            model,
            detached_image,
            candidate["text"],
            geometric_prompt,
            mean,
            std,
            encode_text=True,
        )
        if probe_terms is None:
            continue
        selection_score = compute_prompt_selection_score(
            probe_terms.get("top_logit"),
            probe_terms.get("presence_logit"),
        )
        if selection_score is None:
            continue
        probe_score = float(selection_score.detach().item())
        if best_candidate is None or probe_score > best_score:
            best_candidate = candidate
            best_score = probe_score

    if best_candidate is None:
        best_candidate = candidates[0]

    final_terms = forward_get_toplogit(
        model,
        image_tensor,
        best_candidate["text"],
        geometric_prompt,
        mean,
        std,
        encode_text=True,
    )
    if final_terms is not None:
        final_terms["selected_candidate_index"] = float(best_candidate.get("index", -1.0))
        final_terms["selected_candidate_label"] = str(best_candidate.get("label", ""))
        final_terms["selected_candidate_text"] = str(best_candidate.get("text", ""))
        final_score = compute_prompt_selection_score(
            final_terms.get("top_logit"),
            final_terms.get("presence_logit"),
        )
        final_terms["selection_score"] = float(
            final_score.detach().item() if final_score is not None else best_score
        )
    return best_candidate, final_terms


def compute_pred_cons_presence_losses(
    model: torch.nn.Module,
    adv: torch.Tensor,
    prompt: str,
    prompt_obj: Any,
    prompt_dataset_name: str,
    gt_mask,
    mean,
    std,
    target_pred_logit: torch.Tensor,
    use_pred_logit: bool,
    use_consistency: bool,
    use_presence: bool,
    min_pred_for_update: float,
    w_text: float,
    w_point: float,
    w_box: float,
    w_cons: float,
    w_presence: float,
    prompt_selection_mode: str,
    device: torch.device,
) -> Dict[str, Any]:
    mode = str(prompt_selection_mode).strip().lower()
    if mode not in {"default", "minmax"}:
        raise ValueError(f"Unsupported prompt_selection_mode={prompt_selection_mode}")

    text_terms = None
    point_terms = None
    box_terms = None
    selected_text_prompt = prompt
    text_prompt_choice = 1.0
    point_prompt_choice = -1.0
    box_prompt_choice = -1.0

    if mode == "default":
        text_prompt_struct = build_empty_geometric_prompt(device)
        text_terms = forward_get_toplogit(model, adv, prompt, text_prompt_struct, mean, std, encode_text=True)

        point_entry = build_default_point_prompt_entry(gt_mask, device)
        point_prompt = point_entry["prompt"] if point_entry is not None else None
        if point_prompt is not None:
            point_terms = forward_get_toplogit(model, adv, "visual", point_prompt, mean, std, encode_text=False)
            point_prompt_choice = float(point_entry.get("index", -1.0))

        box_entry = build_default_box_prompt_entry(gt_mask, device)
        box_prompt = box_entry["prompt"] if box_entry is not None else None
        if box_prompt is not None:
            box_terms = forward_get_toplogit(model, adv, "visual", box_prompt, mean, std, encode_text=False)
            box_prompt_choice = float(box_entry.get("index", -1.0))
    else:
        text_candidates = build_text_prompt_candidate_entries(prompt_obj, dataset_name=prompt_dataset_name)
        selected_text_candidate, text_terms = select_best_text_prompt_candidate(
            model=model,
            image_tensor=adv,
            mean=mean,
            std=std,
            candidates=text_candidates,
        )
        if selected_text_candidate is not None:
            selected_text_prompt = str(selected_text_candidate.get("text", prompt))
            text_prompt_choice = float(selected_text_candidate.get("index", -1.0))

        point_candidates = build_point_prompt_candidates_from_mask(gt_mask, device)
        selected_point_candidate, point_terms = select_best_geometric_prompt_candidate(
            model=model,
            image_tensor=adv,
            mean=mean,
            std=std,
            candidates=point_candidates,
        )
        if selected_point_candidate is not None:
            point_prompt_choice = float(selected_point_candidate.get("index", -1.0))

        box_candidates = build_box_prompt_candidates_from_mask(gt_mask, device)
        selected_box_candidate, box_terms = select_best_geometric_prompt_candidate(
            model=model,
            image_tensor=adv,
            mean=mean,
            std=std,
            candidates=box_candidates,
        )
        if selected_box_candidate is not None:
            box_prompt_choice = float(selected_box_candidate.get("index", -1.0))

    out: Dict[str, Any] = {
        "term": None,
        "weighted": 0.0,
        "pred_weighted": 0.0,
        "cons_weighted": 0.0,
        "presence_weighted": 0.0,
        "pred": -1.0,
        "pred_point": -1.0,
        "pred_box": -1.0,
        "pred_score": None,
        "low_pred_sample": False,
        "valid": False,
        "point_prompt_available": int(point_terms is not None),
        "box_prompt_available": int(box_terms is not None),
        "selected_text_prompt": selected_text_prompt,
        "text_prompt_choice": float(text_prompt_choice),
        "point_prompt_choice": float(point_prompt_choice),
        "box_prompt_choice": float(box_prompt_choice),
        "text_prompt_score": -1.0,
        "text_prompt_presence": -1.0,
        "point_prompt_score": -1.0,
        "point_prompt_presence": -1.0,
        "box_prompt_score": -1.0,
        "box_prompt_presence": -1.0,
    }

    if text_terms is None and point_terms is None and box_terms is None:
        return out

    pred_candidates = []
    text_top = text_terms["top_logit"] if text_terms is not None else None
    text_presence = text_terms.get("presence_logit") if text_terms is not None else None
    point_top = point_terms["top_logit"] if point_terms is not None else None
    point_presence = point_terms.get("presence_logit") if point_terms is not None else None
    box_top = box_terms["top_logit"] if box_terms is not None else None
    box_presence = box_terms.get("presence_logit") if box_terms is not None else None

    text_score = text_terms.get("score") if text_terms is not None else None
    if text_score is None:
        text_score = compute_prompt_selection_score(text_top, text_presence)
    point_score = point_terms.get("score") if point_terms is not None else None
    if point_score is None:
        point_score = compute_prompt_selection_score(point_top, point_presence)
    box_score = box_terms.get("score") if box_terms is not None else None
    if box_score is None:
        box_score = compute_prompt_selection_score(box_top, box_presence)

    if text_score is not None:
        pred_candidates.append(text_score)
    if point_score is not None:
        pred_candidates.append(point_score)
    if box_score is not None:
        pred_candidates.append(box_score)

    if len(pred_candidates) == 0:
        return out

    out["valid"] = True
    pred_score = torch.stack(pred_candidates).max()
    out["pred_score"] = pred_score
    low_pred_sample = pred_score.item() < min_pred_for_update
    out["low_pred_sample"] = low_pred_sample

    if text_score is not None:
        out["text_prompt_score"] = float(text_score.detach().item())
    if text_presence is not None:
        out["text_prompt_presence"] = float(text_presence.sigmoid().item())

    if point_score is not None:
        out["point_prompt_score"] = float(point_score.detach().item())
    if point_presence is not None:
        out["point_prompt_presence"] = float(point_presence.sigmoid().item())

    if box_score is not None:
        out["box_prompt_score"] = float(box_score.detach().item())
    if box_presence is not None:
        out["box_prompt_presence"] = float(box_presence.sigmoid().item())

    terms = []

    if use_pred_logit and not low_pred_sample:
        if text_top is not None:
            term = float(w_text) * F.mse_loss(text_top, target_pred_logit.expand_as(text_top))
            terms.append(term)
            out["pred_weighted"] += float(term.detach().item())
            out["pred"] = float(text_score.detach().item()) if text_score is not None else float(text_top.sigmoid().item())
        if point_top is not None:
            term = float(w_point) * F.mse_loss(point_top, target_pred_logit.expand_as(point_top))
            terms.append(term)
            out["pred_weighted"] += float(term.detach().item())
            out["pred_point"] = float(point_score.detach().item()) if point_score is not None else float(point_top.sigmoid().item())
        if box_top is not None:
            term = float(w_box) * F.mse_loss(box_top, target_pred_logit.expand_as(box_top))
            terms.append(term)
            out["pred_weighted"] += float(term.detach().item())
            out["pred_box"] = float(box_score.detach().item()) if box_score is not None else float(box_top.sigmoid().item())

    if use_consistency and not low_pred_sample:
        branch_probs = []
        if text_score is not None:
            branch_probs.append(text_score)
        if point_score is not None:
            branch_probs.append(point_score)
        if box_score is not None:
            branch_probs.append(box_score)
        if len(branch_probs) >= 2:
            pair_losses = []
            for i in range(len(branch_probs)):
                for j in range(i + 1, len(branch_probs)):
                    pair_losses.append(F.mse_loss(branch_probs[i], branch_probs[j]))
            if len(pair_losses) > 0:
                term = float(w_cons) * torch.stack(pair_losses).mean()
                terms.append(term)
                out["cons_weighted"] += float(term.detach().item())

    if use_presence and not low_pred_sample:
        if text_presence is not None:
            term = float(w_presence) * F.mse_loss(text_presence, target_pred_logit.expand_as(text_presence))
            terms.append(term)
            out["presence_weighted"] += float(term.detach().item())
        if point_presence is not None:
            term = float(w_presence) * F.mse_loss(point_presence, target_pred_logit.expand_as(point_presence))
            terms.append(term)
            out["presence_weighted"] += float(term.detach().item())
        if box_presence is not None:
            term = float(w_presence) * F.mse_loss(box_presence, target_pred_logit.expand_as(box_presence))
            terms.append(term)
            out["presence_weighted"] += float(term.detach().item())

    if len(terms) > 0:
        out["term"] = torch.stack(terms).sum()

    out["weighted"] = float(out["pred_weighted"] + out["cons_weighted"] + out["presence_weighted"])
    return out
