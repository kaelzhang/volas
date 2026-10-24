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


def test_pandas_default_backend_examples_distinguish_int_and_bool_na():
    import numpy as np
    import pandas as pd
    from volas import DataFrame

    result = DataFrame({'i': [1, None], 'b': [True, None]}).to_pandas()
    assert result['i'].dtype == np.dtype('float64')
    assert result['i'].iloc[0] == 1 and np.isnan(result['i'].iloc[1])
    assert result['b'].dtype == pd.BooleanDtype()
    assert result['b'].iloc[0] and result['b'].iloc[1] is pd.NA
    for path in (GUIDE, GUIDE.parents[2] / 'docs' / 'interop.md'):
        line, = [line for line in path.read_text().splitlines() if line.startswith('pdf = df.to_pandas()') and ('backend:' in line or '后端：' in line)]
        assert 'int/bool' not in line, path
        assert all(word in line for word in ('int', 'float64', 'NaN', 'bool', 'boolean', 'pd.NA')), path


def test_string_default_and_explicit_object_missing_documentation():
    import numpy as np
    from volas import DataFrame, NA

    frame = DataFrame({'text': ['a', None], 'integer': [1, None]})
    series = frame['text']
    assert series.to_numpy().dtype == np.dtype('O')
    assert series.to_numpy()[1] is None
    assert series.to_numpy(dtype='object')[1] is NA
    assert frame['integer'].to_numpy(dtype='object')[1] is NA
    assert frame.to_numpy(dtype='object')[1, 0] is NA
    assert frame.exec('text')[1] is None
    assert frame.exec('text', dtype='object')[1] is NA
    for path in (GUIDE, GUIDE.parents[2] / 'docs' / 'interop.md',
                 GUIDE.parents[2] / 'docs' / 'series-api.md'):
        text = path.read_text()
        for assertion in ("assert text.to_numpy()[1] is None",
                          "assert text.to_numpy(dtype='object')[1] is NA",
                          "assert integer.to_numpy(dtype='object')[1] is NA"):
            assert assertion in text, path
        block, = [block for block in re.findall(r'```py\n(.*?)```', text, re.S)
                  if "assert text.to_numpy()[1] is None" in block]
        exec(compile(block, str(path), 'exec'), {})
