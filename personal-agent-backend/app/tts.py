"""Compatibility alias for app.services.tts_client."""

import sys
from app.services import tts_client as _module

sys.modules[__name__] = _module
