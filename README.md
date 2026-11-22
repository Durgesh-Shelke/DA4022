# DA4022: Proposal-Aligned Final Evaluation

This directory contains the organized codebase, notebooks, and modular experiment scripts for **DA4022: Test-Time Single-Frame Adaptation for Depth and Geometry Estimation (VGGT Backbone)**.

---

## Directory Overview

```text
DA4022/
├── DA4022_FINAL_KAGGLE.ipynb   # Original notebook configured for Kaggle environments (/kaggle/working)
├── DA4022_LOCAL.ipynb          # Environment-adaptive notebook (auto-detects Kaggle vs local ./workspace)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation and execution instructions
└── code/                       # Standalone modular python scripts
    ├── 01_setup.py             # Environment verification, CAPA repo clone, and deterministic settings
    ├── 02_config.py            # Global experiment configuration, dataset splits, and LR search grid
    ├── 03_toy.py               # Mechanism verification on CPU (quadratic optimization in subspace)
    ├── 04_data.py              # Official ETHZ ScanNet sample scenes download and preparation
    ├── 05_synthetic.py         # Controlled synthetic variations (geometry, camera, texture, lighting)
    ├── 06_engine.py            # VGGT backbone loader, adapter layers (LoRA, bases, hybrid), aligner & loss
    ├── 06b_gpu_preflight.py    # GPU execution gate: 2 updates per method and checkpoint restore test
    ├── 07_donors_bases.py      # Donor adapters training and IRLS robust shared basis extraction
    ├── 08_validation.py        # Validation hyperparameter selection (LR tuning on validation split)
    ├── 09_main.py              # Main benchmark across 7 adaptation methods over test scenes
    ├── 10_robustness.py        # Robustness study (sparse points and simulated measurement noise)
    ├── 11_attributes.py        # Ablation study on individual synthetic attribute bases
    └── 12_report.py            # Result aggregation, metric tables, Markdown/HTML reports, and backup ZIP
```

---

## Pipeline Execution Stages

| Stage | Script | Description | Hardware |
| :--- | :--- | :--- | :--- |
| **1. Setup** | `01_setup.py` | Clones CAPA, sets up deterministic algorithms, verifies environment. | CPU / Network |
| **2. Config** | `02_config.py` | Freezes experiment parameters (`CFG`) and hashes configuration. | CPU |
| **3. Toy** | `03_toy.py` | Validates subspace alignment behavior on synthetic quadratic targets. | CPU |
| **4. Data** | `04_data.py` | Downloads official ScanNet scenes (`scene0707`, `scene0766`, `scene0771`, `scene0777`, `scene0806`). | Network |
| **5. Synthetic** | `05_synthetic.py` | Synthesizes geometric, viewpoint, texture, and illumination variations. | CPU / PIL |
| **6. Engine** | `06_engine.py` | Defines LoRA, Fixed Basis, Random Basis, and Hybrid Residual adapter modules. | GPU |
| **7. Preflight** | `06b_gpu_preflight.py` | Validates forward/backward passes and state checkpointing before long runs. | GPU (1-2 min) |
| **8. Donors & Bases** | `07_donors_bases.py` | Fits donor adapters and extracts IRLS-mined robust subspaces. | GPU |
| **9. Validation** | `08_validation.py` | Evaluates learning rates on validation scene `scene0771` and freezes winners. | GPU |
| **10. Attributes** | `11_attributes.py` | Assesses individual synthetic attribute basis contributions. | GPU |
| **11. Main Test** | `09_main.py` | Runs final test adaptation across methods and seeds on test scenes. | GPU |
| **12. Robustness** | `10_robustness.py` | Stress tests methods against extreme point sparsity and sensor noise. | GPU |
| **13. Reporting** | `12_report.py` | Generates summary tables, convergence plots, and `FINAL_REPORT.html`/`.md`. | CPU |

---

## How to Run

### Method 1: Running on Kaggle (Recommended)
The experiment is configured with a 150-minute GPU budget and requires an active internet connection to download model weights and ScanNet data.
1. Open [Kaggle](https://www.kaggle.com/) and create a new notebook.
2. In the notebook menu, go to **File** -> **Upload Notebook** and upload `DA4022_FINAL_KAGGLE.ipynb`.
3. In the right-hand **Notebook Options** panel:
   - **Accelerator**: Select **GPU T4 x2** or **GPU P100**.
   - **Internet**: Toggle **Internet On**.
4. Run cells sequentially, or click **Run All**.
5. When complete, download `DA4022_FINAL_BACKUP.zip` and the generated report files under `/kaggle/working/DA4022_FINAL_TODAY/report/`.

### Method 2: Running Locally
To run on your local workstation with an NVIDIA GPU:
1. Ensure NVIDIA CUDA drivers and PyTorch with CUDA support are installed:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   pip install -r requirements.txt
   ```
2. Open `DA4022_LOCAL.ipynb` in VS Code or Jupyter Lab.
   - It will automatically set the root workspace to `./workspace/DA4022_FINAL_TODAY` instead of `/kaggle/working`.
3. Execute the cells in order.
