#!/usr/bin/env python3
"""Compress config/persona.md + style/*.md → config/profile_card.md"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.memory.profile import write_profile_card


def main() -> None:
    path = write_profile_card()
    text = path.read_text(encoding="utf-8")
    print(f"Wrote profile card ({len(text)} chars): {path}")


if __name__ == "__main__":
    main()
