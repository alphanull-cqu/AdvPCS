"""Modular universal pred-logit attack for SAM3.

Losses are separated into two parts:
1) temporal memory losses (ptr + memory feature),
2) pred/consistency/presence losses.
"""

import os
import random
import sys
import csv
import numbers
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAM3_ROOT = PROJECT_ROOT / "sam3"
for _p in (PROJECT_ROOT, SAM3_ROOT):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

_LOCAL_SAM2_REPO = PROJECT_ROOT / "sam3" / "sam2"
if (_LOCAL_SAM2_REPO / "sam2" / "build_sam.py").is_file():
    _local_str = str(_LOCAL_SAM2_REPO)
    if _local_str not in sys.path:
        sys.path.insert(0, _local_str)

from sam3.model_builder import build_sam3_video_model
from sam3_util import (
    choose_dataset,
    collate_fn,
    save_image_only,
    seed_everything,
)

from uap_losses_logits import (
    compute_pred_cons_presence_losses,
    compute_target_pred,
    extract_img_id_str,
    extract_video_name_and_frame_idx,
    parse_text_prompt,
    prepare_image,
)
from uap_losses_temporal import compute_temporal_memory_loss, init_temporal_states


class ForceGradContext:
    """Context manager/decorator that forces grad enabled inside wrapped scope."""

    def __enter__(self):
        self.prev_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_grad_enabled(self.prev_grad_enabled)
        return False

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self.__class__():
                return func(*args, **kwargs)

        return wrapper


def parse_args() -> Namespace:
    parser = ArgumentParser(description="UAP on SAM3 pred logits (modular)")
    parser.add_argument("--limit_img", default=100, type=int, help="how many videos (or images) to use; -1 for all")
    parser.add_argument("--limit_frames", default=15, type=int, help="frames per video; -1 for all")
    parser.add_argument("--train_dataset", default="SACO", help="dataset name, e.g., SACO|DAVIS|YOUTUBE|MOSE")
    parser.add_argument("--train_prompts", default="text", help="prompt type, typically text")
    parser.add_argument("--eps", default=10 / 255, type=float, help="L_inf budget")
    parser.add_argument("--alpha", default=2 / 255, type=float, help="PGD step for delta")
    parser.add_argument("--uap_epochs", default=15, type=int, help="how many passes over the dataset")
    parser.add_argument("--log_every", default=20, type=int, help="logging interval (steps)")
    parser.add_argument(
        "--debug_cuda_memory",
        action="store_true",
        help="print lightweight allocated/reserved CUDA memory snapshots around temporal_out on peak or large-jump steps",
    )
    parser.add_argument("--momentum", default=0.95, type=float, help="momentum factor for gradient smoothing")
    parser.add_argument(
        "--attack_mode", default="suppress", choices=["suppress", "hallucinate"], help="target direction"
    )
    parser.add_argument("--target_pred", default=None, type=float, help="optional target pred score in [0,1]")
    parser.add_argument("--w_text", default=1.0, type=float, help="weight for text-prompt branch loss")
    parser.add_argument("--w_geo", default=1.0, type=float, help="deprecated alias for point/box branch weight")
    parser.add_argument("--w_point", default=1.0, type=float, help="weight for point-prompt-only branch loss")
    parser.add_argument("--w_box", default=5.0, type=float, help="weight for box-prompt-only branch loss")
    parser.add_argument("--w_cons", default=0.0, type=float, help="cross-prompt consistency loss weight")
    parser.add_argument(
        "--w_presence",
        default=0.2,
        type=float,
        help="weight for presence-logit branch loss (if model outputs presence logits)",
    )
    parser.add_argument(
        "--w_temporal_ptr",
        default=2.5,
        type=float,
        help="weight for tracker object-pointer temporal disruption loss",
    )
    parser.add_argument(
        "--w_temporal_mem",
        default=2.0,
        type=float,
        help="weight for tracker memory-feature temporal disruption loss",
    )
    parser.add_argument(
        "--ptr_sim_target",
        default=0.0,
        type=float,
        help="hinge target for adjacent-frame object pointer cosine similarity",
    )
    parser.add_argument(
        "--mem_sim_target",
        default=0.0,
        type=float,
        help="hinge target for adjacent-frame memory feature cosine similarity",
    )
    parser.add_argument(
        "--min_pred_for_update",
        default=0.1,
        type=float,
        help="skip pred/pres/cons losses if scores are below this value (temporal still updates)",
    )
    parser.add_argument(
        "--ablation_mode",
        default="all",
        choices=["all", "no_pred", "no_presence", "no_consistency", "no_temporal"],
        help="ablation: drop exactly one loss term (or none with all)",
    )
    parser.add_argument(
        "--prompt_selection_mode",
        default="default",
        choices=["default", "minmax"],
        help="prompt selection strategy for text/point/box branches",
    )
    parser.add_argument("--save_delta", default="predlogit_uap_delta_modular.pt", help="where to store learned delta")
    parser.add_argument(
        "--save_adv", action="store_true", help="save adversarial frames after training the universal perturbation"
    )
    parser.add_argument(
        "--save_root",
        default="sw_video/uap_predlogit_modular",
        help="root folder to dump adversarial frames (will be created if missing)",
    )
    parser.add_argument("--seed", default=30, type=int, help="random seed")

    parser.add_argument(
        "--save_train_log_csv",
        type=str,
        default="",
        help="Optional CSV path for step-wise training metrics; defaults to <save_delta>_epochN_losslog.csv",
    )
    return parser.parse_args()


def append_epoch_suffix(path: str, epoch: int) -> str:
    root, ext = os.path.splitext(path)
    ext = ext if ext else ".pt"
    return f"{root}_epoch{epoch}{ext}"


def _contains_non_finite_prompt(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, str):
        low = x.strip().lower()
        return ("nan" in low) or ("inf" in low)
    if isinstance(x, numbers.Number):
        return not np.isfinite(float(x))
    if torch.is_tensor(x):
        if x.numel() == 0:
            return True
        return not bool(torch.isfinite(x).all().item())
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return True
        return not bool(np.isfinite(x).all())
    if isinstance(x, dict):
        return any(_contains_non_finite_prompt(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return True
        return any(_contains_non_finite_prompt(v) for v in x)

    s = str(x).strip().lower()
    return ("nan" in s) or ("inf" in s)


def save_loss_log_csv(loss_log: List[Dict[str, float]], csv_path: str) -> None:
    if len(loss_log) == 0:
        print(f"[Warn] Empty loss_log, skip writing CSV: {csv_path}")
        return

    fieldnames = sorted({k for row in loss_log for k in row.keys()})
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in loss_log:
            writer.writerow(row)


def _cuda_memory_snapshot(device: torch.device) -> Optional[Dict[str, int]]:
    if not torch.cuda.is_available() or device.type != "cuda":
        return None

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    mib = 1024 * 1024
    return {
        "allocated_mib": int(torch.cuda.memory_allocated(device=device_index) // mib),
        "reserved_mib": int(torch.cuda.memory_reserved(device=device_index) // mib),
        "max_allocated_mib": int(torch.cuda.max_memory_allocated(device=device_index) // mib),
        "max_reserved_mib": int(torch.cuda.max_memory_reserved(device=device_index) // mib),
    }


def _maybe_log_temporal_cuda_memory(
    *,
    debug_enabled: bool,
    epoch: int,
    step: int,
    img_id_str: str,
    use_temporal: bool,
    before: Optional[Dict[str, int]],
    after: Optional[Dict[str, int]],
    temporal_out: Dict[str, Any],
    debug_state: Dict[str, int],
    min_jump_mib: int = 64,
) -> None:
    if not debug_enabled or before is None or after is None:
        return

    delta_alloc = after["allocated_mib"] - before["allocated_mib"]
    delta_reserved = after["reserved_mib"] - before["reserved_mib"]
    peak_alloc = after["max_allocated_mib"] > debug_state["max_allocated_mib"]
    peak_reserved = after["max_reserved_mib"] > debug_state["max_reserved_mib"]
    large_jump = abs(delta_alloc) >= min_jump_mib or abs(delta_reserved) >= min_jump_mib

    if not (peak_alloc or peak_reserved or large_jump):
        return

    debug_state["max_allocated_mib"] = max(debug_state["max_allocated_mib"], after["max_allocated_mib"])
    debug_state["max_reserved_mib"] = max(debug_state["max_reserved_mib"], after["max_reserved_mib"])

    print(
        f"[TemporalMem] epoch={epoch} step={step + 1} img_id={img_id_str} use_temporal={int(use_temporal)} "
        f"term_present={int(temporal_out['term'] is not None)} temp_weighted={float(temporal_out['weighted']):.4f} "
        f"alloc_before_mib={before['allocated_mib']} reserved_before_mib={before['reserved_mib']} "
        f"alloc_after_mib={after['allocated_mib']} reserved_after_mib={after['reserved_mib']} "
        f"d_alloc_mib={delta_alloc} d_reserved_mib={delta_reserved} "
        f"max_alloc_after_mib={after['max_allocated_mib']} max_reserved_after_mib={after['max_reserved_mib']} "
        f"peak_alloc={int(peak_alloc)} peak_reserved={int(peak_reserved)}"
    )


def optimize_uap(
    model: torch.nn.Module,
    data_loader: DataLoader,
    epsilon: float,
    alpha: float,
    epochs: int,
    attack_mode: str,
    target_pred_value: Optional[float],
    log_every: int,
    momentum: float,
    w_text: float,
    w_point: float,
    w_box: float,
    w_cons: float,
    w_presence: float,
    w_temporal_ptr: float,
    w_temporal_mem: float,
    ptr_sim_target: float,
    mem_sim_target: float,
    min_pred_for_update: float,
    ablation_mode: str,
    prompt_selection_mode: str,
    prompt_dataset_name: str = "",
    debug_cuda_memory: bool = False,
) -> Tuple[torch.Tensor, List[Dict[str, float]], List[Dict[str, float]]]:
    device = getattr(model, "device", next(model.parameters()).device)
    image_size = int(getattr(model, "image_size", 1008))
    mean = getattr(model, "image_mean", (0.5, 0.5, 0.5))
    std = getattr(model, "image_std", (0.5, 0.5, 0.5))

    delta = torch.empty(3, image_size, image_size, device=device)
    delta.uniform_(-epsilon, epsilon)
    delta = delta.clamp(-epsilon, epsilon).detach().requires_grad_(True)
    target_pred_logit, tgt_pred_val = compute_target_pred(attack_mode, target_pred_value, device)

    step = 0
    log: List[Dict[str, float]] = []
    epoch_summaries: List[Dict[str, float]] = []
    momentum_grad = torch.zeros_like(delta)
    sample_total_g = torch.zeros_like(delta)
    sample_step_count = 0

    mode = ablation_mode.lower()
    if not str(prompt_dataset_name).strip():
        dataset_obj = getattr(data_loader, "dataset", None)
        dataset_args = getattr(dataset_obj, "args", None)
        prompt_dataset_name = str(getattr(dataset_args, "train_dataset", "") or "")
    use_pred_logit = mode != "no_pred"
    use_presence = mode != "no_presence"
    use_consistency = mode != "no_consistency"
    use_temporal = mode != "no_temporal"
    ptr_sim_target = float(max(-1.0, min(1.0, ptr_sim_target)))
    mem_sim_target = float(max(-1.0, min(1.0, mem_sim_target)))
    if not (use_pred_logit or use_presence or use_consistency or use_temporal):
        raise ValueError(f"Invalid ablation_mode={ablation_mode}; no loss terms enabled")

    print(
        f"[Config] prompt_selection_mode={prompt_selection_mode} | Ablation mode={ablation_mode} | pred={use_pred_logit} "
        f"presence={use_presence} consistency={use_consistency} temporal={use_temporal} "
        f"w_temporal_ptr={float(w_temporal_ptr):.4f} w_temporal_mem={float(w_temporal_mem):.4f} "
        f"ptr_sim_target={ptr_sim_target:.4f} mem_sim_target={mem_sim_target:.4f} "
        f"debug_cuda_memory={int(bool(debug_cuda_memory))}"
    )
    debug_temporal_mem_state = {"max_allocated_mib": 0, "max_reserved_mib": 0}
    for epoch in range(epochs):
        epoch_batches = list(data_loader)
        random.shuffle(epoch_batches)

        video_to_batches: Dict[str, List[Any]] = {}
        for batch in epoch_batches:
            _, _, img_ids_b, _, _ = batch
            img_id_b = extract_img_id_str(img_ids_b)
            video_name_b, _ = extract_video_name_and_frame_idx(img_id_b)
            if video_name_b not in video_to_batches:
                video_to_batches[video_name_b] = []
            video_to_batches[video_name_b].append(batch)

        total_samples = 0
        skipped_low_pred = 0
        skipped_bad_prompt = 0
        point_prompt_available = 0
        box_prompt_available = 0
        temporal_ptr_pairs = 0
        temporal_mem_pairs = 0

        for video_name, video_batches in video_to_batches.items():
            video_batches.sort(key=lambda b: extract_video_name_and_frame_idx(extract_img_id_str(b[2]))[1])
            print(f"[Video Start] epoch={epoch} video={video_name} frames={len(video_batches)}")
            temporal_state = init_temporal_states(num_frames=len(video_batches))

            for images, _, img_ids, gt, point in video_batches:
                prompt_obj = point[0] if isinstance(point, (list, tuple)) and len(point) > 0 else point
                prompt = parse_text_prompt(prompt_obj)
                img = images[0]
                img_t, _, _ = prepare_image(img, image_size, device)

                img_id_str = extract_img_id_str(img_ids)

                if _contains_non_finite_prompt(prompt_obj) or _contains_non_finite_prompt(prompt):
                    skipped_bad_prompt += 1
                    print(
                        f"[Skip-InvalidPrompt] epoch={epoch} step={step+1} img_id={img_id_str} "
                        f"prompt={str(prompt)[:120]}"
                    )
                    step += 1
                    continue

                print(f"[Img Start] epoch={epoch} step={step+1} img_id={img_id_str} prompt={prompt}")

                adv = torch.clamp(img_t + delta, 0.0, 1.0)

                pred_out = compute_pred_cons_presence_losses(
                    model=model,
                    adv=adv,
                    prompt=prompt,
                    prompt_obj=prompt_obj,
                    prompt_dataset_name=str(prompt_dataset_name),
                    gt_mask=gt[0],
                    mean=mean,
                    std=std,
                    target_pred_logit=target_pred_logit,
                    use_pred_logit=use_pred_logit,
                    use_consistency=use_consistency,
                    use_presence=use_presence,
                    min_pred_for_update=float(min_pred_for_update),
                    w_text=float(w_text),
                    w_point=float(w_point),
                    w_box=float(w_box),
                    w_cons=float(w_cons),
                    w_presence=float(w_presence),
                    prompt_selection_mode=str(prompt_selection_mode),
                    device=device,
                )

                if not pred_out["valid"]:
                    step += 1
                    continue

                total_samples += 1
                point_prompt_available += int(pred_out["point_prompt_available"])
                box_prompt_available += int(pred_out["box_prompt_available"])

                if pred_out["low_pred_sample"]:
                    skipped_low_pred += 1
                    print(
                        f"[Skip] epoch={epoch} step={step+1} img_id={img_id_str} "
                        f"pred={pred_out['pred_score'].item():.4f} < {min_pred_for_update:.4f}"
                    )

                if delta.grad is not None:
                    delta.grad.zero_()

                loss = torch.zeros((), device=device, dtype=delta.dtype)

                if pred_out["term"] is not None:
                    loss = loss + pred_out["term"]

                temporal_mem_before = _cuda_memory_snapshot(device) if debug_cuda_memory else None
                temporal_out = compute_temporal_memory_loss(
                    model=model,
                    img_t=img_t,
                    adv=adv,
                    gt_mask=gt[0],
                    image_size=image_size,
                    mean=mean,
                    std=std,
                    temporal_state=temporal_state,
                    use_temporal=use_temporal,
                    w_temporal_ptr=float(w_temporal_ptr),
                    w_temporal_mem=float(w_temporal_mem),
                    ptr_sim_target=float(ptr_sim_target),
                    mem_sim_target=float(mem_sim_target),
                    device=device,
                )
                temporal_mem_after = _cuda_memory_snapshot(device) if debug_cuda_memory else None
                _maybe_log_temporal_cuda_memory(
                    debug_enabled=debug_cuda_memory,
                    epoch=epoch,
                    step=step,
                    img_id_str=img_id_str,
                    use_temporal=use_temporal,
                    before=temporal_mem_before,
                    after=temporal_mem_after,
                    temporal_out=temporal_out,
                    debug_state=debug_temporal_mem_state,
                )
                if temporal_out["term"] is not None:
                    loss = loss + temporal_out["term"]
                temporal_ptr_pairs += int(temporal_out["ptr_pairs"])
                temporal_mem_pairs += int(temporal_out["mem_pairs"])

                if not loss.requires_grad:
                    print(
                        f"[Skip-Backward] epoch={epoch} step={step+1} img_id={img_id_str} "
                        "loss has no grad_fn"
                    )
                    step += 1
                    continue

                loss.backward()
                last_loss = float(loss.item())

                current_grad = delta.grad * loss.detach()
                if epoch == 0:
                    momentum_grad = current_grad.clone()
                else:
                    momentum_grad = momentum * momentum_grad + (1 - momentum) * current_grad

                sample_total_g += momentum_grad
                sample_step_count += 1

                if sample_step_count > 0:
                    avg_gradient = sample_total_g / sample_step_count
                    with torch.no_grad():
                        delta_new = delta - alpha * avg_gradient.sign()
                        delta_new.clamp_(-epsilon, epsilon)
                        delta = delta_new.clone().detach().requires_grad_(True)

                step += 1
                if step % max(1, log_every) == 0:
                    log.append(
                        {
                            "epoch": float(epoch),
                            "step": float(step),
                            "loss": float(last_loss),
                            "pred": float(pred_out["pred"]),
                            "pred_point": float(pred_out["pred_point"]),
                            "pred_box": float(pred_out["pred_box"]),
                            "selected_text_prompt": str(pred_out.get("selected_text_prompt", "")),
                            "text_prompt_choice": float(pred_out.get("text_prompt_choice", -1.0)),
                            "point_prompt_choice": float(pred_out.get("point_prompt_choice", -1.0)),
                            "box_prompt_choice": float(pred_out.get("box_prompt_choice", -1.0)),
                            "temporal_ptr_loss": float(temporal_out["temporal_ptr_loss"]),
                            "temporal_mem_loss": float(temporal_out["temporal_mem_loss"]),
                            "ptr_cos": float(temporal_out["ptr_cos"]),
                            "mem_cos": float(temporal_out["mem_cos"]),
                            "temporal_weighted": float(temporal_out["weighted"]),
                        }
                    )

                print(
                    f"[Img End] epoch={epoch} step={step} img_id={img_id_str} "
                    f"loss={last_loss:.6f} "
                    f"pred_text={pred_out['pred']:.4f} pred_point={pred_out['pred_point']:.4f} "
                    f"pred_box={pred_out['pred_box']:.4f} temp_ptr={temporal_out['temporal_ptr_loss']:.4f} "
                    f"temp_mem={temporal_out['temporal_mem_loss']:.4f} ptr_cos={temporal_out['ptr_cos']:.4f} "
                    f"mem_cos={temporal_out['mem_cos']:.4f} temp_w={temporal_out['weighted']:.4f} "
                    f"txt_choice={pred_out.get('text_prompt_choice', -1.0):.0f} "
                    f"txt={pred_out.get('selected_text_prompt', '')} "
                    f"pt_choice={pred_out.get('point_prompt_choice', -1.0):.0f} "
                    f"bx_choice={pred_out.get('box_prompt_choice', -1.0):.0f}"
                )

                mix_total = max(
                    1e-12,
                    float(pred_out["weighted"]) + float(temporal_out["weighted"]),
                )
                print(
                    f"[LossMix] total={last_loss:.6f} "
                    f"pred_group={float(pred_out['weighted']):.6f}({100.0 * float(pred_out['weighted']) / mix_total:.1f}%) "
                    f"temporal_group={float(temporal_out['weighted']):.6f}({100.0 * float(temporal_out['weighted']) / mix_total:.1f}%) "
                    f"target={tgt_pred_val:.4f}"
                )

            print(f"[Video End] epoch={epoch} video={video_name}")

        print(
            f"[Epoch {epoch} Summary] total_valid={total_samples} "
            f"pred_lt_{min_pred_for_update:.2f}={skipped_low_pred} "
            f"invalid_prompt={skipped_bad_prompt} "
            f"point_prompt_available={point_prompt_available} "
            f"box_prompt_available={box_prompt_available} "
            f"temporal_ptr_pairs={temporal_ptr_pairs} "
            f"temporal_mem_pairs={temporal_mem_pairs}"
        )

        epoch_summaries.append(
            {
                "epoch": float(epoch),
                "total_valid": float(total_samples),
                "pred_lt_min": float(skipped_low_pred),
                "invalid_prompt": float(skipped_bad_prompt),
                "point_prompt_available": float(point_prompt_available),
                "box_prompt_available": float(box_prompt_available),
                "temporal_ptr_pairs": float(temporal_ptr_pairs),
                "temporal_mem_pairs": float(temporal_mem_pairs),
            }
        )

    return delta.detach(), log, epoch_summaries


def save_adversarial_frames(
    data_loader: DataLoader,
    delta: torch.Tensor,
    save_root: str,
    image_size: int,
):
    os.makedirs(save_root, exist_ok=True)
    device = delta.device

    for images, _, img_ids, _, _ in data_loader:
        img = images[0]
        img_id = img_ids[0]
        video_name = img_id.split("/")[0]
        frame_idx = extract_video_name_and_frame_idx(img_id)[1]

        img_t, _, _ = prepare_image(img, image_size, device)
        adv = torch.clamp(img_t + delta, 0.0, 1.0)
        adv_np = (adv.permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
        save_image_only(adv_np, video_name, frame_idx if frame_idx is not None else 0, save_root)


def main():
    args = parse_args()
    seed_everything(seed=args.seed)

    if args.w_point is None:
        args.w_point = float(args.w_geo)
    if args.w_box is None:
        args.w_box = float(args.w_geo)

    custom_dataset = choose_dataset(args)
    data_loader = DataLoader(
        custom_dataset,
        batch_size=1,
        collate_fn=collate_fn,
        num_workers=0,
        shuffle=True,
    )

    orig_no_grad = torch.no_grad
    orig_inference_mode = torch.inference_mode

    with orig_no_grad():
        model = build_sam3_video_model()
    if hasattr(model, "detector") and hasattr(model.detector, "supervise_joint_box_scores"):
        model.detector.supervise_joint_box_scores = False
        print("[Config] Set detector.supervise_joint_box_scores = False for raw pred_logits attack")

    try:
        torch.no_grad = ForceGradContext
        torch.inference_mode = ForceGradContext

        model.eval()
        for p in model.parameters():
            if p.requires_grad:
                p.requires_grad_(False)

        delta, loss_log, epoch_summaries = optimize_uap(
            model=model,
            data_loader=data_loader,
            epsilon=float(args.eps),
            alpha=float(args.alpha),
            epochs=int(args.uap_epochs),
            attack_mode=args.attack_mode,
            target_pred_value=args.target_pred,
            log_every=int(args.log_every),
            momentum=float(args.momentum),
            w_text=float(args.w_text),
            w_point=float(args.w_point),
            w_box=float(args.w_box),
            w_cons=float(args.w_cons),
            w_presence=float(args.w_presence),
            w_temporal_ptr=float(args.w_temporal_ptr),
            w_temporal_mem=float(args.w_temporal_mem),
            ptr_sim_target=float(args.ptr_sim_target),
            mem_sim_target=float(args.mem_sim_target),
            min_pred_for_update=float(args.min_pred_for_update),
            ablation_mode=str(args.ablation_mode),
            prompt_selection_mode=str(args.prompt_selection_mode),
            prompt_dataset_name=str(args.train_dataset),
            debug_cuda_memory=bool(args.debug_cuda_memory),
        )

        save_path = append_epoch_suffix(args.save_delta, args.uap_epochs)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save({"delta": delta.cpu(), "args": vars(args), "loss_log": loss_log}, save_path)
        print(f"Saved universal perturbation to {save_path}")

        txt_path = os.path.splitext(save_path)[0] + ".txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(
                "epoch\ttotal_valid\tpred_lt_min\tinvalid_prompt\tpoint_prompt_available\t"
                "box_prompt_available\ttemporal_ptr_pairs\ttemporal_mem_pairs\n"
            )
            for s in epoch_summaries:
                f.write(
                    f"{int(s['epoch'])}\t{int(s['total_valid'])}\t{int(s['pred_lt_min'])}\t"
                    f"{int(s.get('invalid_prompt', 0.0))}\t"
                    f"{int(s['point_prompt_available'])}\t{int(s['box_prompt_available'])}\t"
                    f"{int(s['temporal_ptr_pairs'])}\t{int(s['temporal_mem_pairs'])}\n"
                )
        print(f"Saved epoch summary to {txt_path}")

        train_log_csv = str(args.save_train_log_csv).strip()
        if not train_log_csv:
            train_log_csv = os.path.splitext(save_path)[0] + "_losslog.csv"
        save_loss_log_csv(loss_log, train_log_csv)
        print(f"Saved training loss log CSV to {train_log_csv}")

        if args.save_adv:
            image_size = int(getattr(model, "image_size", 1008))
            save_adversarial_frames(
                data_loader=data_loader,
                delta=delta,
                save_root=args.save_root,
                image_size=image_size,
            )
            print(f"Adversarial frames saved under {args.save_root}")
    finally:
        torch.no_grad = orig_no_grad
        torch.inference_mode = orig_inference_mode


if __name__ == "__main__":
    main()
