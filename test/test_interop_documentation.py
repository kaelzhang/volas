"""Keep the Chinese interop reference's inline API terms readable."""
from pathlib import Path
import re


GUIDE = Path(__file__).resolve().parents[1] / 'docs' / 'zh-CN' / 'interop.md'


def test_chinese_interop_inline_code_terms_are_separated():
    text = GUIDE.read_text()
    assert not re.search(r'(?<!`)`[^`\n]+``[^`\n]+`(?!`)', text)


def test_chinese_interop_has_one_final_newline():
    text = GUIDE.read_text()
    assert text.endswith('\n') and not text.endswith('\n\n')
