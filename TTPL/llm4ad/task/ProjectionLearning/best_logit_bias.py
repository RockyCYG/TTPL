from __future__ import annotations

import argparse
import os
import sys

import torch


def make_logit_bias(
    coords: torch.Tensor,
    current_node: torch.Tensor,
    ninf_mask: torch.Tensor,
    step: int,
) -> torch.Tensor:
    """Best POMO TSP logit-bias found by the LLM4AD search."""
    batch_size, problem_size, _ = coords.shape
    pomo_size = current_node.shape[1]

    gather_index = current_node[:, :, None].expand(batch_size, pomo_size, 2)
    current_xy = coords.gather(dim=1, index=gather_index)

    dist = torch.cdist(current_xy, coords).clamp_min(1e-6)
    valid_mask = ninf_mask == 0

    topk = min(15, problem_size)
    ranked_dist = dist.masked_fill(~valid_mask, float("inf"))
    topk_index = torch.topk(ranked_dist, k=topk, dim=2, largest=False).indices

    topk_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    topk_mask.scatter_(dim=2, index=topk_index, value=True)

    dist_norm = dist / dist.sum(dim=2, keepdim=True).clamp(min=1e-6)

    local_bias = -torch.log(dist + 1e-6) * (1 - dist_norm)
    global_bias = -dist_norm
    bias = torch.where(topk_mask, local_bias, global_bias)

    exploration_factor = (1 + (step / problem_size) ** 1.5) ** -1
    bias *= exploration_factor

    bias = bias.masked_fill(~valid_mask, 0.0)
    bias = torch.where(torch.isfinite(bias), bias, torch.zeros_like(bias))

    return bias


def run_test(argv: list[str] | None = None) -> tuple[float, float, float]:
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../..")
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from lehd.TSP.test_tsp import add_common_args, main_test

    parser = argparse.ArgumentParser(
        description="Run the final best POMO TSP logit-bias on the test set."
    )
    add_common_args(parser)
    parser.add_argument(
        "--data-path",
        dest="data_path",
        default=None,
        help=(
            "TSP data file or directory. Relative paths are resolved under "
            "lehd/TSP/data; absolute paths are used as-is."
        ),
    )
    args = parser.parse_args(argv)

    if args.data_path is not None:
        args.tsp_data_path = args.data_path

    args.inference_backend = "pomo"
    args.model_load_epoch = 500
    args.pomo_log_dist_bias = False

    score_optimal, score_student, gap = main_test(args, logit_bias=make_logit_bias)
    print(f"Final best logit bias result:")
    print(f"  Teacher SCORE: {score_optimal:.4f}")
    print(f"  Student SCORE: {score_student:.4f}")
    print(f"  Average Gap: {gap * 100:.4f}%")
    return score_optimal, score_student, gap


if __name__ == "__main__":
    run_test()
