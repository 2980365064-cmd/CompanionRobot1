"""Compatibility alias for app.admin.persons."""

import sys
from app.admin import persons as _module

sys.modules[__name__] = _module
