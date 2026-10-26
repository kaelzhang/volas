"""Execute and validate a small computation before publishing release identity."""

import math

from volas import DataFrame

result = DataFrame({"close": [1.0, 2.0, 3.0]}).exec("ma:2")
if len(result) != 3 or not math.isnan(result[0]) or list(result[1:]) != [1.5, 2.5]:
    raise ValueError(f"release smoke computation mismatch: {result}")
