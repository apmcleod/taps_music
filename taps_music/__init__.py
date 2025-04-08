import logging
import sys

stdout_handler = logging.StreamHandler(stream=sys.stdout)
stdout_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(name)s: %(levelname)s >>> %(message)s")
stdout_handler.setFormatter(formatter)

taps_logger = logging.getLogger("taps")
taps_logger.setLevel(logging.INFO)
taps_logger.addHandler(stdout_handler)

from .taps import Taps  # noqa: F401, E402

__version__ = "1.0.0"
