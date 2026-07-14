import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StaticResponsiveTests(unittest.TestCase):
    def test_chat_has_mobile_safe_composer_layout(self):
        html = (ROOT / "static" / "chat.html").read_text(encoding="utf-8")
        self.assertIn("@media (max-width: 640px)", html)
        self.assertIn("env(safe-area-inset-bottom)", html)
        self.assertRegex(html, r"\.composer\s*\{[^}]*position:\s*sticky;")
        self.assertRegex(html, r"#messages\s*\{[^}]*padding:\s*14px\s+12px;")

    def test_admin_has_phone_first_layout_rules(self):
        html = (ROOT / "static" / "admin.html").read_text(encoding="utf-8")
        self.assertIn("@media (max-width: 640px)", html)
        self.assertRegex(html, r"\.sidebar\s*\{[^}]*position:\s*sticky;")
        self.assertRegex(html, r"\.nav\s*\{[^}]*display:\s*flex;")
        self.assertRegex(html, r"#graphCanvas\s*\{[^}]*height:\s*min\(58vh,\s*430px\);")
        self.assertRegex(html, r"\.table-wrap\s*\{[^}]*overflow-x:\s*auto;")


if __name__ == "__main__":
    unittest.main()
