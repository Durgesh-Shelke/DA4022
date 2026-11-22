#!/usr/bin/env python3
import sys
import json
import time
import traceback
import io
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

class Tee(io.TextIOBase):
    def __init__(self, *writers):
        self.writers = writers
    def write(self, s):
        for w in self.writers:
            w.write(s)
            w.flush()
        return len(s)
    def flush(self):
        for w in self.writers:
            w.flush()

def run_notebook(nb_path, output_nb_path=None, stop_after_cell=None):
    nb_path = Path(nb_path)
    if output_nb_path is None:
        output_nb_path = nb_path
    
    with nb_path.open('r', encoding='utf-8') as f:
        nb = json.load(f)
    
    execution_env = {
        '__name__': '__main__',
        '__file__': str(nb_path),
    }

    print(f"Loaded notebook '{nb_path}' with {len(nb['cells'])} cells.")
    print(f"Target execution: up to cell {stop_after_cell if stop_after_cell is not None else len(nb['cells'])-1}")

    for idx, cell in enumerate(nb['cells']):
        if stop_after_cell is not None and idx > stop_after_cell:
            print(f"Stopping execution as requested after cell {stop_after_cell}.")
            break

        if cell.get('cell_type') != 'code':
            continue

        source = ''.join(cell.get('source', []))
        if not source.strip():
            continue

        print(f"\n========================================================")
        print(f"[{time.strftime('%X')}] EXECUTING CELL {idx}")
        print(f"========================================================")
        first_lines = [l for l in source.splitlines() if l.strip() and not l.strip().startswith('#')]
        summary = first_lines[0][:80] if first_lines else "Code cell"
        print(f"Cell {idx} snippet: {summary}")

        buf = io.StringIO()
        tee = Tee(sys.stdout, buf)
        t0 = time.time()
        success = True

        with redirect_stdout(tee), redirect_stderr(tee):
            try:
                exec(source, execution_env)
            except Exception as e:
                success = False
                traceback.print_exc()

        elapsed = time.time() - t0
        output_text = buf.getvalue()
        
        cell['execution_count'] = idx
        cell['outputs'] = [{
            'output_type': 'stream',
            'name': 'stdout' if success else 'stderr',
            'text': output_text.splitlines(keepends=True)
        }]

        # Save progress after every cell
        with Path(output_nb_path).open('w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)

        print(f"[{time.strftime('%X')}] Cell {idx} finished in {elapsed:.2f}s (Success: {success})")
        if not success:
            print(f"Execution halted on cell {idx} due to error.")
            sys.exit(1)

    print("\nNotebook execution finished successfully!")

if __name__ == '__main__':
    nb_file = sys.argv[1] if len(sys.argv) > 1 else '/kaggle/working/DA4022_FINAL_KAGGLE.ipynb'
    stop_idx = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run_notebook(nb_file, stop_after_cell=stop_idx)
