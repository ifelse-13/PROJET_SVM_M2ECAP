import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "generate_quarto_docs.py"
SPEC = importlib.util.spec_from_file_location("generate_quarto_docs", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class GenerateQuartoDocsTests(unittest.TestCase):
    def test_generator_creates_mirrored_docs_and_ignores_binary_and_vendor_dirs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            source_root = Path(tmp_dir) / "repo"
            docs_root = source_root / "docs" / "reference"
            (source_root / "src").mkdir(parents=True)
            (source_root / "node_modules").mkdir()
            (source_root / "src" / "demo.py").write_text(
                "import math\n\n\ndef compute(value):\n    return math.sqrt(value)\n",
                encoding="utf-8",
            )
            (source_root / "README.md").write_text("# Demo\n\nProject summary.\n", encoding="utf-8")
            (source_root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\nbinary")
            (source_root / "node_modules" / "ignored.js").write_text(
                "export const ignored = true;\n",
                encoding="utf-8",
            )

            generated = MODULE.generate_docs(source_root, docs_root)

            demo_doc = docs_root / "src" / "demo.py.qmd"
            readme_doc = docs_root / "readme.md.qmd"
            self.assertIn(demo_doc, generated)
            self.assertIn(readme_doc, generated)
            self.assertFalse((docs_root / "logo.png.qmd").exists())
            self.assertFalse((docs_root / "node_modules").exists())

            demo_content = demo_doc.read_text(encoding="utf-8")
            self.assertIn("Source path: `src/demo.py`", demo_content)
            self.assertIn("functions: compute", demo_content)
            self.assertIn("```python", demo_content)

    def test_generator_creates_directory_indexes_and_notebook_page(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            source_root = Path(tmp_dir) / "repo"
            docs_root = source_root / "docs" / "reference"
            (source_root / "analysis").mkdir(parents=True)
            notebook = {
                "cells": [
                    {"cell_type": "markdown", "metadata": {}, "source": ["# Analysis notebook\n"]},
                    {
                        "cell_type": "code",
                        "execution_count": 1,
                        "metadata": {},
                        "outputs": [],
                        "source": ["import pandas as pd\n", "def train_model():\n", "    return pd.DataFrame()\n"],
                    },
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
            (source_root / "analysis" / "model.ipynb").write_text(json.dumps(notebook), encoding="utf-8")

            MODULE.generate_docs(source_root, docs_root)

            root_index = docs_root / "index.qmd"
            analysis_index = docs_root / "analysis" / "index.qmd"
            notebook_doc = docs_root / "analysis" / "model.ipynb.qmd"

            self.assertTrue(root_index.exists())
            self.assertTrue(analysis_index.exists())
            self.assertTrue(notebook_doc.exists())

            self.assertIn("[analysis](analysis/index.qmd)", root_index.read_text(encoding="utf-8"))
            notebook_content = notebook_doc.read_text(encoding="utf-8")
            self.assertIn("Jupyter notebook with 1 code cells and 1 markdown cells.", notebook_content)
            self.assertIn("functions: train_model", notebook_content)


if __name__ == "__main__":
    unittest.main()
