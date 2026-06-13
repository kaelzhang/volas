"""For ML users: turn OHLCV + indicators into a model-ready tensor.

    pip install torch        # optional; the example degrades gracefully without it
    python examples/05_torch_features.py

volas has no pandas runtime dependency; `to_numpy()` hands you a contiguous
array that feeds NumPy and `torch.Tensor` feature pipelines directly.
"""

import numpy as np

from volas import DataFrame

n = 80
close = 100.0 + np.cumsum(np.random.default_rng(0).normal(0, 0.5, n))
df = DataFrame(
    {
        "open": close - 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.full(n, 1_000.0),
    }
)

# Select a feature matrix: raw columns + cached indicator directives.
features = df[["close", "volume", "rsi:14", "macd", "atr:14"]].to_numpy()
print("feature matrix shape (numpy):", features.shape)

try:
    import torch

    tensor = torch.tensor(features, dtype=torch.float32)
    print("torch tensor shape:", tuple(tensor.shape))
    print(f"OK: built a {tensor.shape[1]}-feature torch tensor from {tensor.shape[0]} bars.")
except ImportError:
    print(f"OK: built a {features.shape[1]}-feature NumPy matrix from {features.shape[0]} bars "
          "(install torch to also build a tensor).")
