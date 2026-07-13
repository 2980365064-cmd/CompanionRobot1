"""Compatibility alias for app.admin.memory."""

import sys
from app.admin import memory as _module

sys.modules[__name__] = _module
