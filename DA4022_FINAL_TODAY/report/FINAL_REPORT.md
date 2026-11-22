# DA4022 final evaluation: measured results

## Objective
Investigate whether fixed bases derived from controlled 3D variations can guide sparse-depth test-time adaptation, and whether a small residual LoRA restores flexibility.

## Data and separation
This is a small custom ScanNet single-frame adaptation study using the official CAPA sample tensors. Donor sequences: scene0707 and scene0766. Validation: scene0771. Test: scene0777 and scene0806. Two frames per sequence are used. No sequence crosses splits. The original sample conditioning is replaced by explicitly seeded random point measurements from valid dense GT. This is not the original CAPA sequence-level protocol or a standard zero-shot benchmark result.

## Controlled synthetic attributes
A simple analytic room/box/sphere renderer generates geometry, camera, texture, and illumination interventions. Within each adapter task, other scene parameters remain fixed and only the named attribute varies; one adapter is shared across the three variants. Two nuisance base scenes are used per attribute. This is low-diversity procedural data, not a reproduction of MegaSynth. Synthetic depth is known from the renderer. Test RGB/GT do not enter basis extraction.

## Methods
Frozen VGGT + sparse-depth alignment; ordinary image-encoder QKV LoRA rank 4; rank-1 LoRA as a near-budget comparator for the hybrid; random fixed bases; pooled synthetic attribute bases; real-donor bases; and synthetic bases + rank-1 residual LoRA. Main learned bases use a rank-d robust shared update mean, approximated by five p=1 IRLS iterations with low-rank QR/SVD factors. Orthonormal singular bases are retained for coefficient optimization. An additional learned_energy method pools Frobenius-normalized uncentered update energy as an extraction ablation. This is a small independent implementation inspired by the shared-update objective, not a full reproduction of the original paper's data and training pipeline. Four-dimensional left/right bases and a trainable 4x4 matrix per layer are used. Combined and individual attribute bases have equal dimensions, but combined bases use more donor tasks. LoRA and residual scaling are 2. All weight decay is zero. The coefficient and residual learning rates may differ; the exact choices are recorded.

## Selection, masks, and timing
Method learning rates are selected using final unobserved-region error on validation frames at a common 30-update budget, then frozen. Main evaluation uses up to 100 updates. GT is used for metrics and generating the declared sparse measurement simulation; no other dense test supervision enters gradients or alignment. Noise multiplies a seeded subset of observed values by uniform[0.2,2.0] factors. All methods share conditioning and corruption for an image/condition. Invalid predictions cause failed runs rather than exclusion. Nonpositive predictions remain in error metrics and their fractions are recorded. No test-best checkpoint is used for final scores.

Timings include synchronized forward/alignment/backward/optimizer work and exclude metrics, file saving, preprocessing and model loading. These are instrumented pilot timings with checkpointing and warm-cache effects, not deployment latency. Offline donor costs are available in the raw measurements. The reported time-to-target is the first recorded checkpoint reaching a predeclared 25% improvement over the aligned baseline; it is not time to a global optimum.

## Interpretation and limits
Tables are generated only from complete runs. Check scene/frame/seed coverage before comparing aggregates. Frames/seeds are not independent scenes. Only two real donor scenes, one validation scene, two test scenes and two synthetic base scenes are used: no broad generalization or statistical significance claim is justified. Seeds vary LoRA initialization and random bases; pure fixed-basis coefficient runs have no random initialization. CUDA reproducibility is best-effort, and warnings/versions are recorded. Attribute overlap is descriptive and does not establish semantic disentanglement. The quadratic experiment deliberately supplies an optimum-containing subspace and demonstrates a mechanism rather than a VGGT guarantee. Historical mid-evaluation numbers are not mixed into these tables. Any accuracy, convergence or robustness benefit must be supported by the generated results, not assumed.

## Stage completion

```json
{
  "donors": "complete",
  "validation": "complete",
  "attribute_ablation": "complete",
  "main": "complete",
  "robustness": "complete"
}
```

Incomplete/failed runs: 0 (see incomplete_runs.json).

## Main coverage (unbalanced partial averages must not be ranked)

```text
{
  "completed_runs": 60,
  "expected_runs": 60,
  "complete": true
}
```

## Main test summary

```text
                mean_scene_absrel_pct  scene_sd  scenes  min_seeds  min_frames_per_scene  mean_adaptation_seconds  trainable_parameters  peak_memory_GB
method                                                                                                                                                 
frozen                       1.909338  0.224239       2          1                     2                 0.000000                   0.0        5.491393
hybrid                       0.938528  0.264404       2          2                     2                79.848567               98688.0        6.080874
learned                      0.939774  0.202374       2          2                     2                79.534481                 384.0        6.079775
learned_energy               0.896714  0.112456       2          2                     2                79.292569                 384.0        6.079775
lora1                        0.954249  0.280941       2          2                     2                79.683500               98304.0        6.079375
lora4                        0.973893  0.254046       2          2                     2                80.034008              393216.0        6.082671
random                       0.989005  0.236564       2          2                     2                79.631807                 384.0        6.079775
shared_real                  0.950607  0.153467       2          2                     2                79.294451                 384.0        6.079775
```

## Robustness (30-update budget; partial coverage explicitly listed)

```text
                      mean_absrel_pct  scenes
method  points noise                         
frozen  20     0.0           1.490264       2
        100    0.0           1.482349       2
               0.1           1.495936       2
               0.2           1.502701       2
        500    0.0           1.472555       2
hybrid  20     0.0           1.190836       2
        100    0.0           1.092826       2
               0.1           1.062349       2
               0.2           1.116371       2
        500    0.0           0.765738       2
learned 20     0.0           1.358369       2
        100    0.0           1.084179       2
               0.1           1.069095       2
               0.2           1.087072       2
        500    0.0           0.834802       2
lora4   20     0.0           1.307678       2
        100    0.0           1.145791       2
               0.1           1.118388       2
               0.2           1.164585       2
        500    0.0           0.728519       2
random  20     0.0           1.178420       2
        100    0.0           1.144189       2
               0.1           1.151342       2
               0.2           1.184197       2
        500    0.0           0.799643       2
```

## Attribute bases: validation diagnostic only

```text
attribute_basis
camera      1.233763
combined    1.246175
geometry    1.074518
lighting    1.267478
texture     1.233670
```

## Presentation sequence
1. Proposal: supervision from CAPA, reusable directions from attribute mining.
2. Controlled known-optimum experiment and its misalignment failure.
3. Actual synthetic interventions, donor/validation/test separation, adapter equations.
4. Validation choices and main test accuracy/parameter/convergence tables.
5. Attribute ablation and learned-versus-random findings, including negative outcomes.
6. Residual-adapter, sparsity and outlier results.
7. Limitations, offline cost, failure cases and conclusions limited to the measured evidence.

## Sources
- CAPA code and license: https://github.com/nv-dvl/capa
- VGGT: https://github.com/facebookresearch/vggt
- Attribute-subspace paper: https://arxiv.org/abs/2604.10095
- Original course proposal and historical findings are maintained separately in the local workspace.
