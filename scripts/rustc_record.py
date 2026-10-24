#!/usr/bin/env python3
"""Record the real executable and argv before handing control to rustc."""

import json
import os
import sys

if __name__ == "__main__":
    # Each JSON line uses a single O_APPEND write, including concurrent crates.
    fd = os.open(
        os.environ["VOLAS_RUSTC_TRACE"], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
    )
    os.write(fd, (json.dumps(sys.argv[1:]) + "\n").encode())
    os.close(fd)
    os.execv(sys.argv[1], sys.argv[1:])
