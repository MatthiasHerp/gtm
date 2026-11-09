"""Make the manuscript's math.tex commands available in notebook MathJax.

The PDF template loads math.tex directly. This script updates the notebook's
otherwise empty first math cell, since MathJax cannot read a local TeX file.
"""
from pathlib import Path
import re

import nbformat

DIRECTORY = Path(__file__).resolve().parent
DEFINITION = re.compile(
    r"\\(newcommand|DeclareMathOperator)(\*)?\s*\{\\([A-Za-z]+)\}\s*\{(.*)\}\s*$"
)


def mathjax_definitions(source):
    definitions = []
    for number, line in enumerate(source.splitlines(), 1):
        line = re.split(r"(?<!\\)%", line, maxsplit=1)[0].strip()
        if not line:
            continue
        match = DEFINITION.fullmatch(line)
        if match is None:
            raise ValueError(f"Unsupported declaration in math.tex at line {number}: {line}")
        kind, star, name, body = match.groups()
        if kind == "DeclareMathOperator":
            body = rf"\operatorname{star or ''}{{{body}}}"
        # MathJax is already in math mode and does not implement TeX page layout.
        # Adapt these formatting commands only in the browser definitions.
        body = body.replace(r"\ensuremath", "").replace(r"\hfill", r"\quad")
        body = re.sub(r"\\text\{\\footnotesize\s+([^{}]*)\}", r"\\text{\1}", body)
        # Global definitions survive the closing math delimiter and can be rerendered.
        definitions.append(rf"\gdef\{name}{{{body}}}")
    return definitions


def sync():
    source = (DIRECTORY / "math.tex").read_text()
    definitions = mathjax_definitions(source)
    path = DIRECTORY / "gtm_sing_scaling.ipynb"
    notebook = nbformat.read(path, as_version=4)
    # Keep individual expressions below MathJax's usual input-buffer limit.
    chunks, current, size = [], [], 0
    for definition in definitions:
        if current and size + len(definition) + 1 > 2800:
            chunks.append(current)
            current, size = [], 0
        current.append(definition)
        size += len(definition) + 1
    if current:
        chunks.append(current)
    cell = nbformat.v4.new_markdown_cell(
        " ".join("$" + " ".join(chunk) + "$" for chunk in chunks))
    cell.id = "gtm-math-macros"
    cell.metadata = {"tags": ["mathjax-macros"], "jupyter": {"source_hidden": True}}
    matching = [i for i, current in enumerate(notebook.cells)
                if "mathjax-macros" in current.metadata.get("tags", [])]
    if len(matching) > 1:
        raise ValueError("Multiple macro definition cells found")
    if matching:
        notebook.cells[matching[0]] = cell
    else:
        notebook.cells.insert(0, cell)
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(f"Synchronized {len(definitions)} macros from {DIRECTORY / 'math.tex'}")


if __name__ == "__main__":
    sync()
