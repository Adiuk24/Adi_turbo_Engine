#!/usr/bin/env python3
"""Calibrate Lloyd-Max codebook for TQ3_S after upstream WHT rotation.

Collects the distribution of values that reach the TQ3_S quantizer
(post-WHT rotation from upstream PR #21038) and computes optimal
8-level Lloyd-Max centroids via iterative algorithm.

Usage:
  # First, dump post-WHT values from a perplexity run:
  GGML_TQ3S_DUMP_DIST=1 ./bin/llama-perplexity -m model.gguf ... -ctk tq3_s

  # Then run this script on the dump:
  python3 scripts/calibrate-tq3s-codebook.py /tmp/tq3s_dist.bin

For now, we compute centroids analytically assuming the post-WHT
distribution is approximately Gaussian (which it is for d>=32).
"""

import numpy as np
from scipy.stats import norm

def lloyd_max_gaussian(n_levels, max_iter=100):
    """Compute optimal Lloyd-Max codebook for standard Gaussian N(0,1)."""
    # Initialize with uniform spacing
    boundaries = np.linspace(-3, 3, n_levels + 1)
    centroids = np.zeros(n_levels)

    for _ in range(max_iter):
        # Update centroids: E[X | b_{i-1} < X < b_i]
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i+1]
            # Conditional expectation of N(0,1) in [lo, hi]
            num = norm.pdf(lo) - norm.pdf(hi)
            den = norm.cdf(hi) - norm.cdf(lo)
            if den > 1e-12:
                centroids[i] = num / den
            else:
                centroids[i] = (lo + hi) / 2

        # Update boundaries: midpoints between centroids
        for i in range(1, n_levels):
            boundaries[i] = (centroids[i-1] + centroids[i]) / 2

    return centroids, boundaries

def lloyd_max_empirical(data, n_levels, max_iter=200):
    """Compute optimal Lloyd-Max codebook from empirical data."""
    # Initialize with quantiles
    percentiles = np.linspace(0, 100, n_levels + 1)
    boundaries = np.percentile(data, percentiles)
    centroids = np.zeros(n_levels)

    for iteration in range(max_iter):
        old_centroids = centroids.copy()

        # Assign each sample to nearest centroid
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i+1]
            mask = (data >= lo) & (data < hi)
            if i == n_levels - 1:
                mask = (data >= lo) & (data <= hi)
            if mask.sum() > 0:
                centroids[i] = data[mask].mean()
            else:
                centroids[i] = (lo + hi) / 2

        # Update boundaries
        for i in range(1, n_levels):
            boundaries[i] = (centroids[i-1] + centroids[i]) / 2

        if np.max(np.abs(centroids - old_centroids)) < 1e-8:
            break

    return centroids, boundaries

if __name__ == "__main__":
    print("=== Lloyd-Max Codebook Calibration for TQ3_S ===\n")

    # Analytical: standard Gaussian (what upstream WHT should produce)
    centroids_gauss, bounds_gauss = lloyd_max_gaussian(8)
    print("Gaussian N(0,1) optimal 8-level codebook:")
    print(f"  Centroids: {{{', '.join(f'{c:.4f}' for c in centroids_gauss)}}}")
    print(f"  Boundaries: {{{', '.join(f'{b:.4f}' for b in bounds_gauss)}}}")
    print()

    # Current hardcoded values in ggml-quants.c
    current = [-2.15, -1.34, -0.76, -0.25, 0.25, 0.76, 1.34, 2.15]
    print(f"Current TQ3_S codebook:")
    print(f"  {{{', '.join(f'{c:.4f}' for c in current)}}}")
    print()

    # Compute MSE for both on Gaussian samples
    np.random.seed(42)
    samples = np.random.randn(1_000_000)

    def quantize_mse(data, centroids):
        # Find nearest centroid for each sample
        indices = np.argmin(np.abs(data[:, None] - centroids[None, :]), axis=1)
        reconstructed = centroids[indices]
        return np.mean((data - reconstructed) ** 2)

    mse_optimal = quantize_mse(samples, centroids_gauss)
    mse_current = quantize_mse(samples, np.array(current))
    mse_uniform = quantize_mse(samples / samples.std() * 2.15,
                                np.linspace(-2.15, 2.15, 8))

    print(f"MSE on N(0,1) samples:")
    print(f"  Optimal Lloyd-Max: {mse_optimal:.6f}")
    print(f"  Current codebook:  {mse_current:.6f}")
    print(f"  Uniform 8-level:   {mse_uniform:.6f}")
    print()

    # For the upstream WHT with head_dim=128, the post-rotation distribution
    # is approximately N(0, sigma/sqrt(d)) where sigma is the pre-rotation RMS.
    # After per-block RMS normalization (sigma = RMS), the normalized values
    # should be approximately N(0, 1/sqrt(d)) ≈ N(0, 0.088) for d=128.
    # But the WHT normalizes by 1/sqrt(block_size) where block_size might
    # differ from head_dim.
    #
    # The key insight: the codebook should be RELATIVE to the per-block scale.
    # The quantizer normalizes by sigma=RMS before looking up centroids.
    # So the codebook should match the distribution of x/sigma, which for
    # Gaussian input is N(0,1) regardless of the actual sigma.

    print("Recommendation:")
    print(f"  Replace hardcoded codebook with:")
    print(f"  static const float cb[8] = {{ {', '.join(f'{c:.4f}f' for c in centroids_gauss)} }};")
