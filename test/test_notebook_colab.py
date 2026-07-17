import json
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _code_cells(path: Path):
    notebook = json.loads(path.read_text())
    return [
        ''.join(cell.get('source', []))
        for cell in notebook.get('cells', [])
        if cell.get('cell_type') == 'code'
    ]


def test_colab_install_cell_does_not_upgrade_numpy():
    cells = _code_cells(ROOT / 'examples/notebooks/volas_quickstart.ipynb')
    install_cells = [
        line
        for cell in cells
        for line in cell.splitlines()
        if line.startswith('%pip install')
    ]

    assert install_cells == ['%pip install -U volas']
    packages = shlex.split(install_cells[0].removeprefix('%pip install').strip())
    assert 'volas' in packages
    assert 'numpy' not in packages
