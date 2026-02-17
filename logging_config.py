from __future__ import annotations

import logging
import os
from typing import Optional


def setup_logging(level: Optional[str] = None) -> None:
    """Configure stdlib logging for the client.

    The GUI already has its own log panel; this config targets console logs
    (useful for debugging when launching from terminal).
    """

    effective_level = (level or os.environ.get("VC_CLIENT_LOG_LEVEL") or os.environ.get("VC_LOG_LEVEL") or "INFO").upper()

    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=effective_level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    else:
        root.setLevel(effective_level)
