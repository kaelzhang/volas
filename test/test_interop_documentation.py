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


def test_arrow_examples_distinguish_default_borrow_and_owned_snapshot():
    root = GUIDE.parents[2]
    for path in (GUIDE, root / 'docs' / 'interop.md', root / 'README.md'):
        text = path.read_text()
        assert "Series.from_arrow(pa_array, name='close', copy=True)" in text, path
        assert not re.search(r"Series\.from_arrow\(pa_array, name='close'\)[^\n]*(?:owned|快照)", text), path
