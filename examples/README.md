# volas examples

Self-contained, runnable scripts — no data files or network needed. Each one
prints an `OK:` line when it succeeds.

```bash
pip install volas
python examples/00_install_check.py
```

Try the browser quickstart:

[![Try the quickstart in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaelzhang/volas/blob/main/examples/notebooks/volas_quickstart.ipynb)
[![Open notebook in GitHub Codespaces](https://img.shields.io/badge/Codespaces-Open%20notebook-181717?logo=githubcodespaces&logoColor=white)](https://codespaces.new/kaelzhang/volas)

| Script | Shows |
| --- | --- |
| [`00_install_check.py`](00_install_check.py) | Verify the install; compute one indicator. |
| [`01_quickstart.py`](01_quickstart.py) | Cache single- and multi-output directives. |
| [`02_pandas_migration.py`](02_pandas_migration.py) | `.iloc` / `.at` / slicing for pandas users. |
| [`03_live_ohlcv_append.py`](03_live_ohlcv_append.py) | **The headline:** append a bar, refresh only the stale tail. |
| [`04_talib_migration.py`](04_talib_migration.py) | RSI / ATR / MACD as directives, for TA-Lib users. |
| [`05_torch_features.py`](05_torch_features.py) | `to_numpy()` → `torch.Tensor` feature matrix. |

Run them all:

```bash
for f in examples/0*.py; do python "$f"; done
```
