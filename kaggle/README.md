# Kaggle execution

This folder is separate from the existing local-GPU notebook and runner at the repository root. The final narrative report is in [../reports/DA4022_Final_Project_Report.md](../reports/DA4022_Final_Project_Report.md), and recorded experiment artifacts are in `../data/DA4022_FINAL_TODAY/`.

## Import the notebook

1. Download [DA4022_FINAL_KAGGLE.ipynb](DA4022_FINAL_KAGGLE.ipynb) and import it into a fresh Kaggle notebook.
2. Enable **Internet** and **GPU**. GPU 0 is used; GPU memory is not pooled.
3. Leave `RESTORE_ZIP` empty for a fresh experiment.
4. Run cells through GPU preflight, check `GPU PREFLIGHT PASSED`, then continue the remaining stages.

The notebook embeds its sources and needs no repository attachment. Old outputs are cleared for distribution; recorded measured results remain in `../data/DA4022_FINAL_TODAY/`.

## Script alternative

Run these cells on Kaggle:

```python
!git clone https://github.com/Durgesh-Shelke/DA4022.git /kaggle/working/DA4022
%run /kaggle/working/DA4022/kaggle/run_kaggle.py --through preflight
```

After preflight passes:

```python
%run /kaggle/working/DA4022/kaggle/run_kaggle.py
```

The runner executes stages in one shared namespace. Do not start the individual stage files as independent processes. A second invocation reuses completed matching runs, although setup, data checks, model loading, and basis extraction may run again.

The script runner reads `code/02_config.py`; the notebook reads its embedded configuration cell. Edit the entry point you actually use before starting. The runner hashes its stage sources to avoid silently mixing different builds. It is a separate executable entry point: use the same entry point and sources when resuming checkpoints.

## Dependencies and output

Setup installs the packages in [requirements.txt](requirements.txt) while retaining Kaggle's existing CUDA-enabled torch/torchvision. No PEFT/TorchAO adapter dependency is needed.

```text
/kaggle/working/capa/                   # Third-party source
/kaggle/working/da4022_downloads/       # Original sample downloads
/kaggle/working/DA4022_FINAL_TODAY/     # Prepared data, checkpoints, results, logs
/kaggle/working/DA4022_FINAL_BACKUP.zip
```

T4 runs use FP16 for transformer computation, FP32 for the depth head, gradient scaling, and block checkpointing.

## Runtime and completeness

The main test alone contains 5,600 updates. The 150-minute default cap is not a promise that all stages finish within that time. Increase `max_gpu_minutes` before execution if the session permits, or resume saved checkpoints. Check stage statuses and scene/seed coverage before drawing conclusions from generated reports.

## Save and resume

Download the backup ZIP generated at stage boundaries and every five completed runs, and save a Kaggle version with outputs. Backups include prepared frames, adaptive checkpoints, code, configuration, and results, but exclude the pretrained backbone and multi-GB original downloads.

For a new session, attach the ZIP, enter its path in the notebook's optional `RESTORE_ZIP` cell, and run that cell before setup. Keep the scientific configuration unchanged. Inspect package-version warnings. Matching completed runs are skipped, and interrupted runs resume from their latest checkpoint.

For the script entry point, restore the project folder to `/kaggle/working/DA4022_FINAL_TODAY/` before invoking the same runner/source version. The script runner attempts partial reporting after an execution error; the notebook's report cell can also be run manually.

## Static validation

From the repository root:

```bash
python kaggle/validate.py
python kaggle/run_kaggle.py --help
```

These commands check syntax, stage order, source consistency, and CLI options without claiming to execute GPU training.
