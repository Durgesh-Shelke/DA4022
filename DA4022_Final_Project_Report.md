# Learning Where to Adapt: Attribute-Guided Test-Time Adaptation for 3D Foundation Models

**DA4022 — Final Project Report**

## Abstract

This project investigates whether reusable adaptation directions can improve sparse-depth completion with a pretrained 3D foundation model. We combine sparse-depth-guided test-time adaptation with fixed subspaces extracted from adapters trained under controlled geometry, camera, texture, and lighting variations. The study includes a known-optimum optimization experiment, synthetic attribute generation, transfer to real ScanNet images, a residual-LoRA extension, and sparsity and outlier evaluations. In the completed main study, the strongest fixed-subspace method achieved **0.8967% unobserved-region absolute relative error**, compared with **0.9739% for ordinary rank-4 LoRA**, while optimizing only **384 rather than 393,216 parameters**. This corresponds to **7.92% lower relative error with 1,024× fewer trainable parameters**. The results establish a promising accuracy–parameter trade-off within the evaluated scenes. They also identify the limits of the approach: benefits depend on the observation regime, combining attributes does not automatically improve performance, and parameter savings do not translate into a substantial measured runtime reduction.

## 1. Research Objective and Project Contribution

Sparse-depth completion requires a model to turn a limited set of depth measurements into a dense prediction. A pretrained 3D foundation model provides a strong geometric prior, but its prediction may still require scene-specific correction. Ordinary LoRA expresses that correction through low-rank factors learned afresh for each input.

Our proposal asks whether useful correction directions can instead be learned beforehand and reused. The project brings together two complementary ideas:

- **CAPA:** sparse observations provide a test-time supervision signal for adapting a frozen foundation model.
- **Attribute-subspace mining:** controlled 3D variations can be used to extract reusable directions for parameter-efficient adaptation.

The central research question is:

> Can attribute-derived fixed subspaces guide adaptation on held-out scenes while preserving prediction quality and reducing the number of parameters optimized at test time?

The project contributes an implemented and evaluated connection between these ideas. It examines both the benefits of restricting adaptation and the consequences of choosing an incomplete or poorly matched subspace. A small residual LoRA is included to test whether scene-specific flexibility can compensate for those restrictions.

## 2. Method

### 2.1 Ordinary and Fixed-Subspace Adaptation

For a frozen projection matrix \(W\), ordinary LoRA learns an update

\[
W' = W + 2BA,
\]

where \(A\) and \(B\) are trainable low-rank factors. The factor of two is the adapter scaling used in the experiments.

Our fixed-subspace formulation instead uses

\[
W' = W + 2UMV^\top,
\]

where \(U\) and \(V\) are precomputed orthonormal bases and only \(M\) is optimized for the target image. The pretrained weights remain frozen. This constrains the correction to previously identified input and output directions.

We adapt the QKV projections in 24 layers of VGGT's image encoder. With four-dimensional left and right bases, each layer requires only a \(4\times4\) coefficient matrix. The total is therefore

\[
24\times4\times4=384
\]

trainable coefficients, compared with 393,216 parameters for ordinary rank-4 LoRA.

### 2.2 Controlled Attribute Data and Basis Extraction

A procedural renderer generates scenes containing room surfaces, a box, and a sphere. It provides exact depth and supports controlled interventions in four attribute families:

| Attribute | Controlled intervention |
|---|---|
| Geometry | Object size and placement |
| Camera | Camera position and resulting viewpoint |
| Texture | Surface-pattern frequency |
| Lighting | Illumination direction |

Two base scenes and three intervention levels per attribute are used. Within each task, the other scene parameters remain fixed, and a single donor adapter is optimized across the three variants. This creates attribute-associated adapter updates without using test-scene depth to extract the bases.

We investigate two extraction strategies:

1. **Robust shared-update bases:** a rank-constrained shared update is estimated using five iteratively reweighted least-squares iterations for a robust objective. Its left and right singular directions define the fixed bases.
2. **Energy-pooled bases:** donor updates are Frobenius-normalized, and their leading left and right update-energy directions are pooled separately. This tests whether retaining variation across donor corrections is more useful than estimating a shared central correction.

Individual attribute bases and combined bases use the same four-dimensional left and right representations. The combined basis has access to more donor tasks, but does not receive a larger test-time coefficient matrix. Shared bases learned from real donor scenes provide an additional comparison.

### 2.3 Residual-LoRA Hybrid

A fixed basis may omit a correction needed by a new scene. To test this limitation, we implement

\[
W' = W + 2UMV^\top + 2B_{\mathrm{res}}A_{\mathrm{res}},
\]

with a rank-1 residual adapter. The hybrid has 98,688 trainable parameters. Ordinary rank-1 LoRA, with 98,304 parameters, is included as a nearly matched parameter-budget control.

### 2.4 Sparse Supervision and Evaluation

At test time, scale and shift are estimated from the supplied sparse measurements, and adaptation minimizes the depth error at those measurement locations. Dense test depth is used for evaluation, not as an additional optimization target. Sparse measurements are simulated by sampling valid ground-truth pixels; artificial corruption is applied only to the declared observed subset.

The primary metric is **absolute relative error on valid pixels outside the conditioning set**, reported as a percentage:

\[
\operatorname{AbsRel}(\%)=
\frac{100}{|\Omega_{\mathrm{unobs}}|}
\sum_{i\in\Omega_{\mathrm{unobs}}}
\frac{|\widehat D_i-D_i|}{D_i}.
\]

This evaluates the completion itself, rather than merely rewarding agreement with points already supplied to the model. Lower values are better throughout this report.

## 3. Experimental Design

### 3.1 Separation of Learning, Selection, and Testing

| Role | ScanNet sequences | Purpose |
|---|---|---|
| Real donors | `scene0707`, `scene0766` | Learn reusable real-scene update directions |
| Validation | `scene0771` | Select learning rates and diagnose attribute bases |
| Test | `scene0777`, `scene0806` | Evaluate the selected methods on held-out sequences |

No sequence crosses these groups. Here, “held-out” refers to the project's donor and validation data, rather than a claim about VGGT's original pretraining data.

Learning rates are selected on validation data at a common 30-update budget and frozen before the main comparison. Main test scores are evaluated at the fixed final budget, rather than selecting the best checkpoint using test ground truth.

### 3.2 Evaluation Protocols

| Experiment | Evaluation inputs | Budget and conditions |
|---|---|---|
| Main comparison | Two frames from each of two test sequences; two seeds for adapted methods | 100 updates; 100 clean depth points |
| Sparsity | First selected frame from each test sequence; one seed | 30 updates; 20, 100, or 500 points |
| Outliers | Same two-frame subset as the sparsity study; one seed | 30 updates; 100 points with 0%, 10%, or 20% corruption |
| Attribute ablation | Two validation frames | 30 updates; equal basis dimensions |

For outliers, selected observed values are multiplied by factors sampled uniformly from \([0.2,2.0]\). Methods share the measurement locations and corruption for a given input and condition. Consequently, robustness comparisons are paired within the tested conditions.

The main study completed **all 60 planned runs**: 56 adaptation runs across seven methods, four frames, and two seeds, together with four frozen-baseline predictions. The final experiment summary reports **no incomplete or failed runs**. Frame results are averaged within each sequence before computing the mean across test sequences.

## 4. Controlled Optimization: When Does a Constraint Preserve the Optimum?

Before evaluating a nonlinear foundation model, we studied a positive-definite quadratic with a known optimum. This separates the effect of the search space from uncertainty about the best attainable solution.

The experiment uses 60 ambient parameters, a six-dimensional constrained representation, ten seeds, and a maximum of 1,500 gradient-descent updates. Each method uses the same curvature-based rule for choosing its constant learning rate. The convergence target is a normalized objective gap of \(10^{-6}\).

| Search space | Runs reaching the target | Median updates when reached | Median effective condition number |
|---|---:|---:|---:|
| Full space | 0/10 within 1,500 updates | Not reached within budget | 10,000.00 |
| Aligned subspace containing the optimum | **10/10** | **25** | **8.71** |
| Misaligned subspace | 0/10 | Not reached | 9.64 |
| Random subspace | 0/10 | Not reached | 5.10 |

The aligned subspace preserves the optimum while improving the conditioning of the optimization problem. Misaligned and random subspaces can be well-conditioned yet converge to inferior constrained solutions: their median best attainable normalized gaps are approximately 0.205 and 0.687, respectively.

**The key result is that good conditioning and useful representational capacity must be considered together.** A small search space is beneficial only if it retains the correction required by the task. The full-space optimum remains attainable; it was not reached within the tested iteration budget. This deliberately constructed experiment explains the mechanism behind the proposal, rather than establishing a global-optimum guarantee for VGGT.

## 5. Main Result: Better Mean Accuracy with 1,024× Fewer Trainable Parameters

The final real-image comparison shows that reusable fixed bases can support effective adaptation outside the observed pixels.

| Method | Mean unobserved AbsRel (%) ↓ | Trainable parameters | Reduction relative to rank-4 LoRA |
|---|---:|---:|---:|
| Aligned frozen VGGT | 1.9093 | 0 | — |
| Ordinary rank-4 LoRA | 0.9739 | 393,216 | Reference |
| Ordinary rank-1 LoRA | 0.9542 | 98,304 | 4× fewer |
| Random fixed bases | 0.9890 | 384 | 1,024× fewer |
| Real-donor shared bases | 0.9506 | 384 | 1,024× fewer |
| Synthetic robust shared-update bases | 0.9398 | 384 | 1,024× fewer |
| Synthetic bases + residual LoRA | 0.9385 | 98,688 | Approximately 3.98× fewer |
| **Synthetic energy-pooled bases** | **0.8967** | **384** | **1,024× fewer** |

### 5.1 The Strongest Accuracy–Parameter Trade-Off

The energy-pooled method achieves the lowest mean error among the evaluated methods. Relative to ordinary rank-4 LoRA, it reduces error from 0.973893% to 0.896714%—a **7.92% relative reduction**, or 0.0772 percentage points—while requiring **99.90% fewer trainable parameters**. Relative to the aligned frozen model, its error reduction is **53.04%**.

These results directly support the project's central premise: useful scene corrections need not require learning all low-rank factors anew. A small set of coefficients over previously learned directions can provide a strong adaptation space in the evaluated setting.

### 5.2 Learned Directions Add Value Beyond Restricting Capacity

The random-basis control uses the same 384-parameter budget as the learned fixed-basis methods. Energy-pooled bases achieve **9.33% lower mean error than random bases**. This provides evidence, within this study, that the choice of directions matters beyond the effect of reducing the number of trainable variables.

Synthetic robust bases and real-donor bases also improve on the random-basis mean. The comparison therefore supports investigating reusable directions as an informative prior, while the small test set limits the strength of any generalization claim.

### 5.3 Extraction Strategy Matters

Energy pooling performs better than the robust shared-update extraction in the main comparison: 0.8967% versus 0.9398%. A plausible interpretation is that preserving several useful variations across donor updates can be more effective than representing their shared central correction. The experiment establishes the observed ranking; it does not isolate that explanation as the sole cause.

### 5.4 What the Hybrid Adds

The hybrid achieves 0.9385% error, compared with 0.9542% for the nearly parameter-matched rank-1 LoRA. This is a **1.65% relative error reduction** in the evaluated scenes and suggests that reusable directions can complement a small freely learned residual.

However, its main-study advantage over pure robust fixed bases is only **0.00125 percentage points**, despite a substantially larger parameter budget. Thus, the hybrid is better justified as a flexibility mechanism for particular conditions than as the most parameter-efficient main configuration.

## 6. Sparse Measurements: The Preferred Adaptation Space Depends on Available Evidence

The sparsity study uses a separate, common 30-update budget and the first selected frame from each test sequence. Its values should not be compared directly with the 100-update main-study scores.

| Method | 20 points | 100 points | 500 points |
|---|---:|---:|---:|
| Aligned frozen VGGT | 1.4903 | 1.4823 | 1.4726 |
| Ordinary rank-4 LoRA | 1.3077 | 1.1458 | **0.7285** |
| Synthetic robust bases | 1.3584 | **1.0842** | 0.8348 |
| Random fixed bases | **1.1784** | 1.1442 | 0.7996 |
| Synthetic bases + residual LoRA | 1.1908 | 1.0928 | 0.7657 |

Three complementary findings emerge:

- **At 100 points, learned fixed bases provide the lowest error among the tested methods.** This is a useful regime in which reusable corrections can compensate for limited scene-specific supervision.
- **At 20 points, random bases perform best.** The learned fixed bases do not provide a universal advantage under extreme sparsity. Adding the residual improves over pure learned bases by **12.33% relative error**, although random bases remain slightly better.
- **At 500 points, ordinary LoRA performs best.** With more measurements, a more flexible adaptation space can exploit additional evidence that the restricted basis may not express.

The result is a conditional account of parameter efficiency: the most suitable adaptation space depends on both the quality of the prior directions and the amount of target-scene evidence.

## 7. Outlier Robustness: Stable Learned-Basis Performance in the Tested Corruption Regime

The outlier experiment holds the point budget at 100 and uses 30 updates for every adapted method.

| Method | 0% corruption | 10% corruption | 20% corruption |
|---|---:|---:|---:|
| Aligned frozen VGGT | 1.4823 | 1.4959 | 1.5027 |
| Ordinary rank-4 LoRA | 1.1458 | 1.1184 | 1.1646 |
| **Synthetic robust bases** | **1.0842** | 1.0691 | **1.0871** |
| Random fixed bases | 1.1442 | 1.1513 | 1.1842 |
| Synthetic bases + residual LoRA | 1.0928 | **1.0623** | 1.1164 |

Pure learned bases maintain nearly unchanged error between the clean and 20%-corrupted settings: 1.084179% and 1.087072%, respectively. At 20% corruption, they achieve **6.66% lower relative error than ordinary LoRA** and the lowest mean error among the tested methods.

This is encouraging evidence of stability under the particular corruption model. Some mildly corrupted settings also score better than their clean counterparts; with two scenes and one corruption seed, this should be treated as variation in the small experiment, not evidence that adding noise is beneficial.

## 8. Attribute Ablation: Relevant Directions Matter More Than Combining Everything

Individual attribute bases are evaluated on validation data using equal basis dimensions and a common update budget.

| Basis source | Validation unobserved AbsRel (%) ↓ |
|---|---:|
| **Geometry** | **1.0745** |
| Texture | 1.2337 |
| Camera | 1.2338 |
| Combined attributes | 1.2462 |
| Lighting | 1.2675 |

Geometry-derived bases give the strongest validation result, with **13.77% lower relative error than the combined basis**. This is consistent with the geometric nature of depth completion, but does not establish that geometry bases would be best on an independent test set.

The combined basis does not automatically dominate individual attributes. Under a fixed four-dimensional capacity, merging information from additional attributes may dilute directions most relevant to the task. This motivates task-aware basis selection or adaptive allocation of basis dimensions. Such extensions are motivated by the findings rather than demonstrated outcomes of the current project.

## 9. Efficiency: A Clear Parameter Saving, Not a Claimed Runtime Breakthrough

| Method | Mean adaptation time (s) | Peak GPU memory (GB) |
|---|---:|---:|
| Ordinary rank-4 LoRA | 80.03 | 6.083 |
| Synthetic robust bases | 79.53 | 6.080 |
| Synthetic energy-pooled bases | 79.29 | 6.080 |
| Synthetic bases + residual LoRA | 79.85 | 6.081 |

Despite the large reduction in trainable variables, the measured adaptation time and peak memory are similar. Forward and backward computation through the foundation model still dominate the workload. The less-than-one-percent timing difference between energy-pooled bases and ordinary LoRA is not treated as a reliable practical speedup.

The demonstrated efficiency contribution is therefore **a much smaller trainable adaptation state with improved mean accuracy**, not a proportional reduction in total GPU memory or inference latency. Timings exclude model loading, preprocessing, metric calculation, file saving, and offline donor training. Faster real-model convergence also cannot be established from the final error table alone; it requires a matched accuracy-target analysis of the recorded curves.

## 10. Conclusions Against the Original Proposal

| Proposed objective | Conclusion supported by the experiments |
|---|---|
| Reuse adaptation directions on new project-held-out scenes | Supported within the small synthetic-to-real and real-donor study |
| Reduce test-time trainable parameters | Strongly supported: 384 coefficients versus 393,216 LoRA parameters |
| Preserve or improve completion accuracy | Supported in the main mean results; energy pooling gives the strongest result |
| Improve behavior with sparse observations | Condition-dependent; learned bases lead at 100 points but not at 20 or 500 |
| Improve outlier robustness | Encouraging results for learned bases under the tested corruption model |
| Preserve flexibility through a residual LoRA | Supported in selected sparse conditions; little main-accuracy gain over pure fixed bases |
| Reach an optimum through a constrained space | Demonstrated in the constructed quadratic when the optimum lies in the subspace |
| Reduce real-model convergence time or runtime | Not established by the reported final scores and timing measurements |

The project establishes a meaningful bridge between reusable attribute-related adaptation directions and scene-specific sparse-depth supervision. Its strongest result is the **accuracy–parameter trade-off**: energy-pooled synthetic bases achieve the best mean test error while reducing the trainable adaptation state by three orders of magnitude.

Equally important, the experiments explain where the idea needs refinement. Learned directions must match the target correction; a fixed basis can be too restrictive when more observations become available; and combining attributes is not automatically better than selecting a relevant one. The residual adapter offers a practical way to regain flexibility, but its additional capacity must be justified by the operating condition.

**The final conclusion is that learning where to adapt is a promising way to make adaptation more parameter-efficient, provided the subspace and its flexibility are matched to the task.** The evidence supports this conclusion for the evaluated scenes without asserting universal superiority, semantic disentanglement, or guaranteed global-optimum convergence in a foundation model.

## 11. Scope and Next Steps

This is a controlled course-project study with two real donor sequences, one validation sequence, two test sequences, and two procedural base scenes. Repeated frames and seeds do not substitute for additional independent scenes. The results establish feasibility and useful comparative evidence, rather than statistical significance or benchmark-wide performance.

For pure fixed-basis methods, the coefficient matrices start at zero for both seeds; these repeats do not represent independently sampled learned bases. Random-basis and ordinary-LoRA seeds affect their respective initializations.

The most directly motivated extensions are broader donor diversity, additional held-out scenes, repeated corruption realizations, task-aware attribute selection, and adaptive basis dimensions. A larger study should also compare time to a common accuracy target and account for the offline cost of constructing the bases.

## References

1. *Depth Completion as Parameter-Efficient Test-Time Adaptation* (CAPA). [Paper](https://arxiv.org/abs/2602.14751) · [Implementation](https://github.com/nv-dvl/capa).
2. *Mining Attribute Subspaces for Efficient Fine-tuning of 3D Foundation Models*. [Paper](https://arxiv.org/abs/2604.10095).
3. *VGGT: Visual Geometry Grounded Transformer*. [Implementation](https://github.com/facebookresearch/vggt).

---

*Evidence basis: the completed final-evaluation summary supplied on 24 November 2026 and the earlier controlled-quadratic experiment. Main, robustness, validation, and controlled-optimization results are kept separate. Earlier incomplete main-study averages are excluded.*
