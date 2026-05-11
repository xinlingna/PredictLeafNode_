"""
export_compat_wrapper.py

Export a TorchScript-compatible wrapper that accepts the v1 C++ input format:
    input: [1, K+1, D]  (row 0 = query, rows 1..K = centroids, all unnormalized)

Internally the wrapper:
  1. Normalizes input using training-data mean/std (baked into the model)
  2. Splits query [1,D] and centroids [1,K,D]
  3. Computes L2 distances and builds centroid_feat [1,K,D+1]
  4. Calls the v2 model forward(query, centroid_feat)

The output .pt file is a drop-in replacement for the existing ELPIS model file
with no C++ code changes required.

Usage:
    python3 export_compat_wrapper.py \
        --scripted_pt  <path>/_scripted_fp32.pt \
        --train_npz    <path>/train_deep50M.npz \
        --centroids_path <path>/centroids.npy \
        [--output      <path>/_scripted_compat_fp32.pt]
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class V2CompatWrapper(nn.Module):
    """
    Wraps a v2 QueryCentroidModel to accept the v1 input format [1, K+1, D].

    The first row of the input is the raw query; rows 1..K are the raw centroids.
    Normalization is applied internally using training-data statistics.
    """
    def __init__(self,
                 inner: torch.jit.ScriptModule,
                 mean: torch.Tensor,
                 std: torch.Tensor) -> None:
        super().__init__()
        self.inner = inner
        # Register as buffers so they move with the module (CPU/CUDA) and are
        # included in the saved TorchScript file.
        self.register_buffer("mean", mean)   # [1, D]
        self.register_buffer("std",  std)    # [1, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: float32 tensor of shape [1, K+1, D]
               Row 0 is the query vector; rows 1..K are centroid vectors.
               Values are in the original (unnormalized) coordinate space.
        Returns:
            logits: float32 tensor of shape [1, K]
        """
        q = x[:, 0, :]    # [1, D]  raw query
        C = x[:, 1:, :]   # [1, K, D]  raw centroids

        mean = self.mean                       # [1, D]
        std  = self.std                        # [1, D]

        q_n = (q - mean) / std                 # [1, D]   normalized query
        C_n = (C - mean.unsqueeze(1)) / std.unsqueeze(1)  # [1, K, D]

        diff     = C_n - q_n.unsqueeze(1)      # [1, K, D]
        dist     = diff.norm(dim=-1, keepdim=True)   # [1, K, 1]
        cent_feat = torch.cat([C_n, dist], dim=-1)   # [1, K, D+1]

        return self.inner(q_n, cent_feat)      # [1, K]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Export v2 compat wrapper for v1 C++ API")
    p.add_argument("--scripted_pt",     required=True,
                   help="Path to _scripted_fp32.pt produced by training")
    p.add_argument("--train_npz",       required=True,
                   help="Training NPZ (queries + targets) used to compute norm stats")
    p.add_argument("--centroids_path",  required=True,
                   help="centroids.npy file")
    p.add_argument("--output",          default=None,
                   help="Output path (default: replaces '_scripted_fp32' with '_scripted_compat_fp32')")
    args = p.parse_args()

    # ── Output path ──────────────────────────────────────────────────────────
    if args.output is None:
        args.output = args.scripted_pt.replace("_scripted_fp32.pt",
                                               "_scripted_compat_fp32.pt")
        if args.output == args.scripted_pt:
            args.output = args.scripted_pt.replace(".pt", "_compat.pt")

    # ── Compute normalization stats from training data ───────────────────────
    print("Computing normalization stats from training data...")
    data = np.load(args.train_npz)
    queries = data["queries"].astype(np.float32)      # [N, D]
    centroids = np.load(args.centroids_path).astype(np.float32)  # [K, D]

    flat = np.concatenate([queries, centroids], axis=0)  # [N+K, D]
    mean_np = flat.mean(axis=0, keepdims=True).astype(np.float32)  # [1, D]
    std_np  = (flat.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)

    mean_t = torch.from_numpy(mean_np)
    std_t  = torch.from_numpy(std_np)
    print(f"  mean range: [{mean_np.min():.4f}, {mean_np.max():.4f}]")
    print(f"  std  range: [{std_np.min():.6f},  {std_np.max():.4f}]")
    del data, queries, flat  # free RAM

    # ── Load inner scripted model ─────────────────────────────────────────────
    print(f"Loading inner model from {args.scripted_pt} ...")
    inner = torch.jit.load(args.scripted_pt, map_location="cpu")
    inner.eval()

    # ── Build and trace wrapper ───────────────────────────────────────────────
    K = centroids.shape[0]
    D = centroids.shape[1]
    print(f"K={K}, D={D}")

    wrapper = V2CompatWrapper(inner, mean_t, std_t)
    wrapper.eval()

    # Use torch.jit.trace with a representative input (no dynamic control flow)
    example = torch.randn(1, K + 1, D)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)

    # Verify output shape matches
    with torch.no_grad():
        out = traced(example)
    assert out.shape == (1, K), f"Unexpected output shape: {out.shape}"
    print(f"Output shape verified: {out.shape}")

    # ── Save ─────────────────────────────────────────────────────────────────
    traced.save(args.output)
    print(f"Saved compat wrapper → {args.output}")
    print()
    print("In ELPIS, point --model_file to the new .pt file.")
    print("No C++ changes required — the wrapper accepts the same [1,K+1,D] input.")


if __name__ == "__main__":
    main()
