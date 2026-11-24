# DA4022: Learning Where to Adapt

Attribute-guided test-time adaptation for depth completion with a frozen VGGT backbone. The project combines sparse-depth supervision with reusable subspaces and evaluates ordinary LoRA, fixed bases, and a residual-LoRA hybrid.

**[Read the final project report](DA4022_Final_Project_Report.md)** for the findings and conclusions. The report is at the repository root, outside both execution workflows.

## Main result

Synthetic energy-pooled bases achieved **0.8967% unobserved-region AbsRel**, compared with **0.9739% for rank-4 LoRA**, while training **384 rather than 393,216 parameters**: **7.92% lower relative error with 1,024x fewer trainable parameters**.

This is a small, two-test-scene study: the result demonstrates an accuracy–parameter trade-off, not statistically established broad superiority or a proportional runtime speedup. The committed environment records an **NVIDIA GeForce RTX 4060 Laptop GPU**. Those measured timings should not be presented as Kaggle T4 timings.

## Repository layout

```text
DA4022/
|-- DA4022_Final_Project_Report.md  # Final findings and conclusions
|-- README.md
|-- DA4022_LOCAL.ipynb             # Existing local-GPU notebook
|-- run_notebook.py                # Existing notebook execution helper
|-- requirements.txt               # Local environment dependencies
|-- code/                         # Existing modular experiment stages
|-- kaggle/                       # Separate Kaggle execution code
|   |-- DA4022_FINAL_KAGGLE.ipynb  # Self-contained Kaggle notebook
|   |-- run_kaggle.py              # Script runner with shared stage state
|   |-- code/                     # Matching Kaggle-specific stage sources
|   |-- requirements.txt          # Extra packages; keeps Kaggle's PyTorch
|   |-- validate.py               # Syntax and notebook/source checks
|   `-- README.md                 # Kaggle instructions
`-- DA4022_FINAL_TODAY/            # Recorded results and provenance
    |-- report/                   # Tables, figures, generated summaries
    |-- environment.json
    |-- config.json
    `-- selected_settings.json
```

The local-GPU entry points retain their existing paths. The Kaggle notebook has moved from the root into `kaggle/`. `DA4022_FINAL_TODAY/code/` is the recorded source snapshot, not another execution entry point.

## Local NVIDIA GPU

1. Clone this repository and enter its root directory.
2. Install CUDA-enabled PyTorch and torchvision appropriate for your GPU driver using the [official installation selector](https://pytorch.org/get-started/locally/).
3. Install the remaining dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Open [DA4022_LOCAL.ipynb](DA4022_LOCAL.ipynb) in Jupyter or VS Code and execute cells in order. Outside Kaggle, it uses `./workspace/` relative to the notebook's working directory.

Alternatively, from the repository root:

```bash
python run_notebook.py DA4022_LOCAL.ipynb
```

The existing helper saves execution outputs back into the notebook it runs; use a copy if you want to keep an unexecuted version.

## Kaggle GPU

Import **[kaggle/DA4022_FINAL_KAGGLE.ipynb](kaggle/DA4022_FINAL_KAGGLE.ipynb)** into a fresh Kaggle notebook. Enable **Internet** and **GPU**, leave `RESTORE_ZIP` empty for a fresh experiment, and run cells in order. The notebook downloads its sample data and pretrained weights; previous notebook state is not required.

Run through **GPU preflight** first. Continue the longer stages after `GPU PREFLIGHT PASSED`. The pipeline uses GPU 0 only; two GPUs do not pool memory. T4 execution uses FP16 with gradient scaling and an FP32 depth head.

See **[kaggle/README.md](kaggle/README.md)** for the script alternative, backups, and resume instructions.

## Experiment stages

1. Setup and configuration.
2. Known-optimum quadratic experiment.
3. Scene-disjoint data preparation.
4. Controlled geometry, camera, texture, and lighting generation.
5. VGGT loading and GPU preflight.
6. Donor adaptation and robust/energy-pooled basis extraction.
7. Validation-only learning-rate selection.
8. Individual-attribute ablation.
9. Main test across seven adaptation methods.
10. Sparse-depth and outlier studies.
11. Tables, convergence figures, reports, and backup ZIP.

Main testing uses 100 updates, two frames per test scene, and two adaptation seeds. Robustness uses 30 updates, the first selected frame per test scene, and one seed. Do not compare their scores as if the protocols were identical.

## Runtime, configuration, and recovery

- Edit scientific settings before a fresh run: the notebook's configuration cell, or `kaggle/code/02_config.py` for the script runner. Use a new result directory for changed scientific settings.
- The default main study alone contains **5,600 optimizer updates**, in addition to donors, validation, ablations, and robustness. Runtime depends on GPU and network performance.
- The **150-minute default GPU-stage wall-time cap may stop the experiment before completion**. Set an appropriate `max_gpu_minutes` or resume saved checkpoints; a budget stop is not a completed evaluation.
- Matching completed runs are reused. Interrupted runs can resume from their latest 10-update checkpoint.
- Download backups and save Kaggle versions with outputs. Files left only in `/kaggle/working` are not durable.

## Results and provenance

- [Final project report](DA4022_Final_Project_Report.md)
- [Main-study coverage](DA4022_FINAL_TODAY/report/main_coverage.json)
- [Main results](DA4022_FINAL_TODAY/report/main_summary.csv)
- [Per-scene/per-seed results](DA4022_FINAL_TODAY/report/main_scene_seed.csv)
- [Robustness results](DA4022_FINAL_TODAY/report/robustness_summary.csv)
- [Recorded environment](DA4022_FINAL_TODAY/environment.json)
- [Frozen validation choices](DA4022_FINAL_TODAY/selected_settings.json)

Large downloads, prepared frames, tensor checkpoints, and learned tensor files are excluded by `.gitignore`. Committed summaries, figures, configurations, and source snapshots document the completed evaluation. Fresh execution writes a separate working copy.

## Foundations

- [CAPA](https://github.com/nv-dvl/capa)
- [Mining Attribute Subspaces for Efficient Fine-tuning of 3D Foundation Models](https://arxiv.org/abs/2604.10095)
- [VGGT](https://github.com/facebookresearch/vggt)

Third-party code and pretrained models retain their respective licenses and terms.
