"""Temporal-memory losses for UAP attack.

This module keeps pointer-step and memory-step temporal disruption terms.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


def norm_by_mean_std(img_01: torch.Tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=img_01.device, dtype=img_01.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, device=img_01.device, dtype=img_01.dtype).view(3, 1, 1)
    return (img_01 - mean_t) / std_t


def pool_maskmem_feature(maskmem_feat: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if maskmem_feat is None or not torch.is_tensor(maskmem_feat):
        return None
    if maskmem_feat.dim() != 4:
        return None
    return maskmem_feat.mean(dim=(-1, -2))


def detach_tracker_out_dict(tracker_out: Dict[str, Any]) -> Dict[str, Any]:
    detached: Dict[str, Any] = {}
    for key, value in tracker_out.items():
        if torch.is_tensor(value):
            detached[key] = value.detach()
        elif isinstance(value, list):
            detached[key] = [x.detach() if torch.is_tensor(x) else x for x in value]
        else:
            detached[key] = value
    return detached


def tracker_forward_temporal_tokens(
    model: torch.nn.Module,
    adv_01: torch.Tensor,
    point_inputs,
    mean,
    std,
    tracker_state: Dict[str, Any],
):
    if not hasattr(model, "tracker") or model.tracker is None:
        return None

    tracker = model.tracker
    try:
        if adv_01.dim() == 3:
            adv_01 = adv_01.unsqueeze(0)
        adv_norm = norm_by_mean_std(adv_01, mean=mean, std=std)

        visual_backbone = getattr(tracker, "backbone", None)
        if visual_backbone is None and hasattr(model, "detector"):
            visual_backbone = getattr(model.detector, "backbone", None)
        if visual_backbone is None or not hasattr(visual_backbone, "forward_image"):
            return None

        backbone_raw = visual_backbone.forward_image(adv_norm)
        backbone_out = backbone_raw.get("sam2_backbone_out", backbone_raw) if isinstance(backbone_raw, dict) else backbone_raw

        if (
            isinstance(backbone_out, dict)
            and "backbone_fpn" in backbone_out
            and len(backbone_out["backbone_fpn"]) >= 2
            and hasattr(tracker, "sam_mask_decoder")
        ):
            backbone_out = backbone_out.copy()
            backbone_out["backbone_fpn"][0] = tracker.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
            backbone_out["backbone_fpn"][1] = tracker.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])

        _, vision_feats, vision_pos_embeds, feat_sizes = tracker._prepare_backbone_features(backbone_out)

        frame_idx = int(tracker_state.get("frame_idx", 0))
        num_frames = int(tracker_state.get("num_frames", 2))
        is_init_cond_frame = frame_idx == 0

        current_out = tracker.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos_embeds,
            feat_sizes=feat_sizes,
            image=adv_norm,
            point_inputs=point_inputs,
            mask_inputs=None,
            output_dict=tracker_state["output_dict"],
            num_frames=num_frames,
            run_mem_encoder=True,
            use_prev_mem_frame=True,
        )

        stored_out = detach_tracker_out_dict(current_out)
        if is_init_cond_frame:
            tracker_state["output_dict"]["cond_frame_outputs"][frame_idx] = stored_out
        else:
            tracker_state["output_dict"]["non_cond_frame_outputs"][frame_idx] = stored_out
        tracker_state["frame_idx"] = frame_idx + 1

        return {
            "obj_ptr": current_out.get("obj_ptr", None),
            "maskmem_feat": current_out.get("maskmem_features", None),
        }
    except Exception as exc:
        print(f"Error in tracker temporal forward: {exc}")
        return None


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

    px = float(xs.mean()) / denom_w
    py = float(ys.mean()) / denom_h
    px = float(np.clip(px, 1e-6, 1.0 - 1e-6))
    py = float(np.clip(py, 1e-6, 1.0 - 1e-6))
    return px, py


def build_tracker_point_inputs_from_mask(mask_data: Any, image_size: int, device: torch.device):
    point_xy = extract_point_box_from_mask(mask_data)
    if point_xy is None:
        return None
    px, py = point_xy
    x = float(np.clip(px * float(image_size), 0.0, float(image_size - 1)))
    y = float(np.clip(py * float(image_size), 0.0, float(image_size - 1)))
    point_coords = torch.tensor([[[x, y]]], device=device, dtype=torch.float32)
    point_labels = torch.ones((1, 1), device=device, dtype=torch.int32)
    return {"point_coords": point_coords, "point_labels": point_labels}


def init_temporal_states(num_frames: int) -> Dict[str, Any]:
    return {
        "tracker_adv": {
            "frame_idx": 0,
            "num_frames": num_frames,
            "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
        },
        "tracker_clean": {
            "frame_idx": 0,
            "num_frames": num_frames,
            "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
        },
        "prev_obj_ptr_adv": None,
        "prev_obj_ptr_clean": None,
        "prev_mem_pool_adv": None,
        "prev_mem_pool_clean": None,
    }


def _clone_tensor_or_none(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().clone()
    return x


def _clone_tracker_state(tracker_state: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "frame_idx": int(tracker_state.get("frame_idx", 0)),
        "num_frames": int(tracker_state.get("num_frames", 2)),
        "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
    }
    for k, v in tracker_state.get("output_dict", {}).get("cond_frame_outputs", {}).items():
        out["output_dict"]["cond_frame_outputs"][k] = detach_tracker_out_dict(v)
    for k, v in tracker_state.get("output_dict", {}).get("non_cond_frame_outputs", {}).items():
        out["output_dict"]["non_cond_frame_outputs"][k] = detach_tracker_out_dict(v)
    return out


def clone_temporal_states(temporal_state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tracker_adv": _clone_tracker_state(temporal_state["tracker_adv"]),
        "tracker_clean": _clone_tracker_state(temporal_state["tracker_clean"]),
        "prev_obj_ptr_adv": _clone_tensor_or_none(temporal_state.get("prev_obj_ptr_adv", None)),
        "prev_obj_ptr_clean": _clone_tensor_or_none(temporal_state.get("prev_obj_ptr_clean", None)),
        "prev_mem_pool_adv": _clone_tensor_or_none(temporal_state.get("prev_mem_pool_adv", None)),
        "prev_mem_pool_clean": _clone_tensor_or_none(temporal_state.get("prev_mem_pool_clean", None)),
    }


def advance_temporal_states_with_clean_frame(
    model: torch.nn.Module,
    img_t: torch.Tensor,
    gt_mask,
    image_size: int,
    mean,
    std,
    temporal_state: Dict[str, Any],
    device: torch.device,
) -> None:
    """Advance temporal states once per real frame using clean image only.

    This keeps temporal progression aligned with real frame order, while avoiding
    local-step-induced state drift inside per-frame PGD optimization.
    """
    tracker_point_inputs = build_tracker_point_inputs_from_mask(gt_mask, image_size, device)

    clean_terms_clean = tracker_forward_temporal_tokens(
        model=model,
        adv_01=img_t,
        point_inputs=tracker_point_inputs,
        mean=mean,
        std=std,
        tracker_state=temporal_state["tracker_clean"],
    )
    clean_terms_adv = tracker_forward_temporal_tokens(
        model=model,
        adv_01=img_t,
        point_inputs=tracker_point_inputs,
        mean=mean,
        std=std,
        tracker_state=temporal_state["tracker_adv"],
    )

    ref_terms = clean_terms_clean if clean_terms_clean is not None else clean_terms_adv
    if ref_terms is None:
        return

    obj_ptr = ref_terms.get("obj_ptr", None)
    mem_pool = pool_maskmem_feature(ref_terms.get("maskmem_feat", None))

    if obj_ptr is not None:
        obj_ptr_d = obj_ptr.detach()
        temporal_state["prev_obj_ptr_clean"] = obj_ptr_d
        temporal_state["prev_obj_ptr_adv"] = obj_ptr_d
    if mem_pool is not None:
        mem_pool_d = mem_pool.detach()
        temporal_state["prev_mem_pool_clean"] = mem_pool_d
        temporal_state["prev_mem_pool_adv"] = mem_pool_d


def compute_temporal_memory_loss(
    model: torch.nn.Module,
    img_t: torch.Tensor,
    adv: torch.Tensor,
    gt_mask,
    image_size: int,
    mean,
    std,
    temporal_state: Dict[str, Any],
    use_temporal: bool,
    w_temporal_ptr: float,
    w_temporal_mem: float,
    ptr_sim_target: float,
    mem_sim_target: float,
    device: torch.device,
) -> Dict[str, float]:
    out = {
        "term": None,
        "weighted": 0.0,
        "temporal_ptr_loss": -1.0,
        "temporal_mem_loss": -1.0,
        "ptr_cos": -1.0,
        "mem_cos": -1.0,
        "ptr_pairs": 0,
        "mem_pairs": 0,
    }

    tracker_point_inputs = build_tracker_point_inputs_from_mask(gt_mask, image_size, device)
    tracker_terms_adv = tracker_forward_temporal_tokens(
        model=model,
        adv_01=adv,
        point_inputs=tracker_point_inputs,
        mean=mean,
        std=std,
        tracker_state=temporal_state["tracker_adv"],
    )
    with torch.no_grad():
        tracker_terms_clean = tracker_forward_temporal_tokens(
            model=model,
            adv_01=img_t,
            point_inputs=tracker_point_inputs,
            mean=mean,
            std=std,
            tracker_state=temporal_state["tracker_clean"],
        )

    if not use_temporal or tracker_terms_adv is None or tracker_terms_clean is None:
        return out

    terms = []

    if float(w_temporal_ptr) > 0.0:
        cur_obj_ptr = tracker_terms_adv.get("obj_ptr", None)
        cur_obj_ptr_clean = tracker_terms_clean.get("obj_ptr", None)
        prev_obj_ptr = temporal_state["prev_obj_ptr_adv"]
        prev_obj_ptr_clean = temporal_state["prev_obj_ptr_clean"]

        if (
            cur_obj_ptr is not None
            and prev_obj_ptr is not None
            and cur_obj_ptr_clean is not None
            and prev_obj_ptr_clean is not None
        ):
            adv_ptr_step = cur_obj_ptr - prev_obj_ptr
            clean_ptr_step = (cur_obj_ptr_clean - prev_obj_ptr_clean).detach()
            ptr_cos = F.cosine_similarity(adv_ptr_step, clean_ptr_step, dim=-1)
            ptr_target = torch.full_like(ptr_cos, ptr_sim_target)
            temporal_ptr_term = F.relu(ptr_cos - ptr_target).mean()
            terms.append(float(w_temporal_ptr) * temporal_ptr_term)
            out["temporal_ptr_loss"] = float(temporal_ptr_term.item())
            out["ptr_cos"] = float(ptr_cos.mean().item())
            out["weighted"] += float(w_temporal_ptr) * out["temporal_ptr_loss"]
            out["ptr_pairs"] = int(ptr_cos.numel())

    if float(w_temporal_mem) > 0.0:
        cur_mem_pool = pool_maskmem_feature(tracker_terms_adv.get("maskmem_feat", None))
        cur_mem_pool_clean = pool_maskmem_feature(tracker_terms_clean.get("maskmem_feat", None))
        prev_mem_pool = temporal_state["prev_mem_pool_adv"]
        prev_mem_pool_clean = temporal_state["prev_mem_pool_clean"]

        if (
            cur_mem_pool is not None
            and prev_mem_pool is not None
            and cur_mem_pool_clean is not None
            and prev_mem_pool_clean is not None
        ):
            adv_mem_step = cur_mem_pool - prev_mem_pool
            clean_mem_step = (cur_mem_pool_clean - prev_mem_pool_clean).detach()
            mem_cos = F.cosine_similarity(adv_mem_step, clean_mem_step, dim=-1)
            mem_target = torch.full_like(mem_cos, mem_sim_target)
            temporal_mem_term = F.relu(mem_cos - mem_target).mean()
            terms.append(float(w_temporal_mem) * temporal_mem_term)
            out["temporal_mem_loss"] = float(temporal_mem_term.item())
            out["mem_cos"] = float(mem_cos.mean().item())
            out["weighted"] += float(w_temporal_mem) * out["temporal_mem_loss"]
            out["mem_pairs"] = int(mem_cos.numel())

    if tracker_terms_adv.get("obj_ptr") is not None:
        temporal_state["prev_obj_ptr_adv"] = tracker_terms_adv["obj_ptr"].detach()
    if tracker_terms_clean.get("obj_ptr") is not None:
        temporal_state["prev_obj_ptr_clean"] = tracker_terms_clean["obj_ptr"].detach()

    cur_mem_adv = pool_maskmem_feature(tracker_terms_adv.get("maskmem_feat", None))
    cur_mem_clean = pool_maskmem_feature(tracker_terms_clean.get("maskmem_feat", None))
    if cur_mem_adv is not None:
        temporal_state["prev_mem_pool_adv"] = cur_mem_adv.detach()
    if cur_mem_clean is not None:
        temporal_state["prev_mem_pool_clean"] = cur_mem_clean.detach()

    if len(terms) > 0:
        out["term"] = torch.stack(terms).sum()
    return out
