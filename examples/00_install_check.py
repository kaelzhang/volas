"""Verify that volas installed correctly and can compute an indicator.

    pip install volas
    python examples/00_install_check.py
"""

from volas import DataFrame

print("volas import OK")

df = DataFrame(
    {
        "open": [1.0, 2.0, 3.0, 4.0, 5.0],
        "high": [2.0, 3.0, 4.0, 5.0, 6.0],
        "low": [0.5, 1.5, 2.5, 3.5, 4.5],
        "close": [1.5, 2.5, 3.5, 4.5, 5.5],
        "volume": [100, 120, 130, 140, 150],
    }
)

# A directive (`ma:2`) is computed once and cached as a column on the frame.
print(df["ma:2"])

print("OK: volas is installed and computed a 2-period moving average.")
