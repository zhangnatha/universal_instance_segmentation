#!/usr/bin/env python3
"""用于 convert_to_labelme 的便捷入口脚本。"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from instance_segmentation.data.convert_to_labelme import main

if __name__ == "__main__":
    main()
