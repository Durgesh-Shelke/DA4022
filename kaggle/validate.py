#!/usr/bin/env python3
"""Check syntax and source/notebook consistency without GPU dependencies."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from run_kaggle import STAGES


def main() -> None:
    folder = Path(__file__).resolve().parent
    files = {name: folder / "code" / name for _, name in STAGES}
    for path in [*files.values(), folder / "run_kaggle.py", folder / "validate.py"]:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    notebook = json.loads((folder / "DA4022_FINAL_KAGGLE.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    embedded = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compile(source, f"notebook_cell_{index}", "exec")
        assert not cell.get("outputs"), "Distributed notebook contains old outputs"
        for statement in ast.parse(source).body:
            if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
                continue
            call = statement.value
            if not (isinstance(call.func, ast.Attribute) and call.func.attr == "write_text"
                    and call.args and isinstance(call.args[0], ast.Constant)
                    and isinstance(call.args[0].value, str)):
                continue
            target = call.func.value
            if not (isinstance(target, ast.BinOp) and isinstance(target.right, ast.Constant)):
                continue
            filename = target.right.value
            if filename in files:
                assert call.args[0].value == files[filename].read_text(encoding="utf-8"), filename
                embedded.append(filename)
    assert embedded == [filename for _, filename in STAGES], "Notebook/source order differs"
    assert len(STAGES) == 13
    assert (folder.parent / "DA4022_Final_Project_Report.md").is_file()
    print("PASS: 13 stages compile and match embedded notebook sources in order.")
    print("PASS: notebook cells compile, outputs are cleared, root report exists.")
    print("GPU training was not run by this validation command.")


if __name__ == "__main__":
    main()
