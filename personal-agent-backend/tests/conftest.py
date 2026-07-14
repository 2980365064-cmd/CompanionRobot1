"""测试共享 fixtures"""
from pathlib import Path

import pytest


@pytest.fixture
def root_dir() -> Path:
    """返回项目根目录（personal-agent-backend/）"""
    return Path(__file__).resolve().parents[1]
