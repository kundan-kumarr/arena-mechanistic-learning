# ARENA Mechanistic Learning
**100 Days of Mechanistic Learning — ARENA + Notes, Implementations, Experiments & Blogs**

##100DaysofMechanisticLearning

*Started: March 1, 2026 · Target completion: June 9, 2026*

Based on: [ARENA – Alignment Research Engineer Accelerator](https://learn.arena.education/)

[![Days Complete](https://img.shields.io/badge/Days%20Complete-0%2F100-red?style=flat-square)](./daily_notes/)
[![Module](https://img.shields.io/badge/Current%20Module-0%3A%20Fundamentals-blue?style=flat-square)](./01_fundamentals/)
[![Blog Posts](https://img.shields.io/badge/Blog%20Posts-0-orange?style=flat-square)](./blog_posts/)

---

## Overview

This repository is my **public learning log** for the [ARENA (Alignment Research Engineer Accelerator)](https://learn.arena.education/) curriculum — a structured path into AI safety research through mechanistic interpretability, reinforcement learning, and alignment science.

It contains:

- **Personal notes** — concepts explained in my own words
- **Mathematical derivations** — where intuition meets formalism
- **Implementations from scratch** — NumPy / minimal PyTorch, readable over clever
- **Experiments & sanity checks** — gradient checks, ablations, visualizations
- **Blog posts** — one per day, compressing each concept into a clear narrative

The goal is **mechanistic understanding** — not just running notebooks, but knowing *why* every piece works.

---

## 100 Days of Mechanistic Learning Challenge

Every day I:
1. Study one concept from the ARENA curriculum
2. Implement it from scratch (minimal, readable code)
3. Run at least one check or experiment
4. Write a blog post that compresses the idea into a clear narrative

**This is not a claim of novel research. It is a public record of learning, implementation, and reflection.**

My daily workflow:

```
Read → Derive → Implement → Verify → Experiment → Write
```

---

## Progress Tracker

| Module | Days | Topic | Status |
|---|---|---|---|
| 0 | 1–28 | Fundamentals (PyTorch → VAEs & GANs) | Started |
| 1 | 29–65 | Mechanistic Interpretability | Not started |
| 2 | 66–76 | Reinforcement Learning + RLHF | Not started |
| 3 | 77–86 | LLM Evaluations | Not started |
| 4 | 87–100 | Alignment Science | Not started |

---

## Daily Log

### MODULE 0 — Fundamentals
**ARENA Chapter 0 · Days 1–28**

<details>
<summary><strong>Week 1 — Prerequisites & PyTorch Fluency (Days 1–5)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 1 | PyTorch Tensors & Basics | [0.0 Prerequisites](https://learn.arena.education/chapter0_fundamentals/00_prereqs/) | [notes](./daily_notes/day_001.md) | [code](./00_prerequisites/day_001_tensors.py) | [blog](./blog_posts/day_001.md) |
| 2 | einops & einsum | [0.0 Prerequisites](https://learn.arena.education/chapter0_fundamentals/00_prereqs/) | [notes](./daily_notes/day_002.md) | [code](./00_prerequisites/day_002_einops.py) | [blog](./blog_posts/day_002.md) |
| 3 | Tensor Manipulation Patterns | [0.0 Prerequisites](https://learn.arena.education/chapter0_fundamentals/00_prereqs/) | [notes](./daily_notes/day_003.md) | [code](./00_prerequisites/day_003_patterns.py) | [blog](./blog_posts/day_003.md) |
| 4 | Linear Algebra in Code | [0.0 Prerequisites](https://learn.arena.education/chapter0_fundamentals/00_prereqs/) | [notes](./daily_notes/day_004.md) | [code](./00_prerequisites/day_004_linalg.py) | [blog](./blog_posts/day_004.md) |
| 5 | Review + Speed Drills | [0.0 Prerequisites](https://learn.arena.education/chapter0_fundamentals/00_prereqs/) | [notes](./daily_notes/day_005.md) | [code](./00_prerequisites/day_005_review.py) | [blog](./blog_posts/day_005.md) |

</details>

<details>
<summary><strong>Week 2 — Ray Tracing & Batched Operations (Days 6–10)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 6 | Ray Tracing Intro | [0.1 Ray Tracing](https://learn.arena.education/chapter0_fundamentals/01_ray_tracing/) | [notes](./daily_notes/day_006.md) | [code](./01_ray_tracing/day_006_intro.py) | [blog](./blog_posts/day_006.md) |
| 7 | Batched Rays | [0.1 Ray Tracing](https://learn.arena.education/chapter0_fundamentals/01_ray_tracing/) | [notes](./daily_notes/day_007.md) | [code](./01_ray_tracing/day_007_batched.py) | [blog](./blog_posts/day_007.md) |
| 8 | Triangle Meshes | [0.1 Ray Tracing](https://learn.arena.education/chapter0_fundamentals/01_ray_tracing/) | [notes](./daily_notes/day_008.md) | [code](./01_ray_tracing/day_008_meshes.py) | [blog](./blog_posts/day_008.md) |
| 9 | Full Render Pipeline | [0.1 Ray Tracing](https://learn.arena.education/chapter0_fundamentals/01_ray_tracing/) | [notes](./daily_notes/day_009.md) | [code](./01_ray_tracing/day_009_render.py) | [blog](./blog_posts/day_009.md) |
| 10 | Performance Optimization | [0.1 Ray Tracing](https://learn.arena.education/chapter0_fundamentals/01_ray_tracing/) | [notes](./daily_notes/day_010.md) | [code](./01_ray_tracing/day_010_perf.py) | [blog](./blog_posts/day_010.md) |

</details>

<details>
<summary><strong>Week 3 — CNNs & ResNets (Days 11–14)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 11 | Conv Operations from Scratch | [0.2 CNNs & ResNets](https://learn.arena.education/chapter0_fundamentals/02_cnns/) | [notes](./daily_notes/day_011.md) | [code](./02_cnns_resnets/day_011_conv.py) | [blog](./blog_posts/day_011.md) |
| 12 | MNIST Classifier | [0.2 CNNs & ResNets](https://learn.arena.education/chapter0_fundamentals/02_cnns/) | [notes](./daily_notes/day_012.md) | [code](./02_cnns_resnets/day_012_mnist.py) | [blog](./blog_posts/day_012.md) |
| 13 | ResNets — Theory & Architecture | [0.2 CNNs & ResNets](https://learn.arena.education/chapter0_fundamentals/02_cnns/) | [notes](./daily_notes/day_013.md) | [code](./02_cnns_resnets/day_013_resnet_arch.py) | [blog](./blog_posts/day_013.md) |
| 14 | ResNet Training on CIFAR-10 | [0.2 CNNs & ResNets](https://learn.arena.education/chapter0_fundamentals/02_cnns/) | [notes](./daily_notes/day_014.md) | [code](./02_cnns_resnets/day_014_cifar.py) | [blog](./blog_posts/day_014.md) |

</details>

<details>
<summary><strong>Week 3–4 — Optimization (Days 15–19)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 15 | SGD from Scratch | [0.3 Optimization](https://learn.arena.education/chapter0_fundamentals/03_optimization/) | [notes](./daily_notes/day_015.md) | [code](./03_optimization/day_015_sgd.py) | [blog](./blog_posts/day_015.md) |
| 16 | Adam Optimizer | [0.3 Optimization](https://learn.arena.education/chapter0_fundamentals/03_optimization/) | [notes](./daily_notes/day_016.md) | [code](./03_optimization/day_016_adam.py) | [blog](./blog_posts/day_016.md) |
| 17 | Weights & Biases Setup | [0.3 Optimization](https://learn.arena.education/chapter0_fundamentals/03_optimization/) | [notes](./daily_notes/day_017.md) | [code](./03_optimization/day_017_wandb.py) | [blog](./blog_posts/day_017.md) |
| 18 | Hyperparameter Search | [0.3 Optimization](https://learn.arena.education/chapter0_fundamentals/03_optimization/) | [notes](./daily_notes/day_018.md) | [code](./03_optimization/day_018_sweeps.py) | [blog](./blog_posts/day_018.md) |
| 19 | Loss Landscape Visualization | [0.3 Optimization](https://learn.arena.education/chapter0_fundamentals/03_optimization/) | [notes](./daily_notes/day_019.md) | [code](./03_optimization/day_019_landscapes.py) | [blog](./blog_posts/day_019.md) |

</details>

<details>
<summary><strong>Week 4 — Backpropagation from Scratch (Days 20–24)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 20 | Autograd Theory + Value class | [0.4 Backpropagation](https://learn.arena.education/chapter0_fundamentals/04_backprop/) | [notes](./daily_notes/day_020.md) | [code](./04_backprop/day_020_value.py) | [blog](./blog_posts/day_020.md) |
| 21 | Chain Rule in Code | [0.4 Backpropagation](https://learn.arena.education/chapter0_fundamentals/04_backprop/) | [notes](./daily_notes/day_021.md) | [code](./04_backprop/day_021_chain_rule.py) | [blog](./blog_posts/day_021.md) |
| 22 | Custom Backward Passes | [0.4 Backpropagation](https://learn.arena.education/chapter0_fundamentals/04_backprop/) | [notes](./daily_notes/day_022.md) | [code](./04_backprop/day_022_custom_backward.py) | [blog](./blog_posts/day_022.md) |
| 23 | Train MLP with Custom Autograd | [0.4 Backpropagation](https://learn.arena.education/chapter0_fundamentals/04_backprop/) | [notes](./daily_notes/day_023.md) | [code](./04_backprop/day_023_mlp_autograd.py) | [blog](./blog_posts/day_023.md) |
| 24 | Gradient Checking | [0.4 Backpropagation](https://learn.arena.education/chapter0_fundamentals/04_backprop/) | [notes](./daily_notes/day_024.md) | [code](./04_backprop/day_024_grad_check.py) | [blog](./blog_posts/day_024.md) |

</details>

<details>
<summary><strong>Week 4–5 — VAEs & GANs (Days 25–28)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 25 | Autoencoders + VAE Theory | [0.5 VAEs & GANs](https://learn.arena.education/chapter0_fundamentals/05_vaes_gans/) | [notes](./daily_notes/day_025.md) | [code](./05_vaes_gans/day_025_ae.py) | [blog](./blog_posts/day_025.md) |
| 26 | VAE Training + ELBO | [0.5 VAEs & GANs](https://learn.arena.education/chapter0_fundamentals/05_vaes_gans/) | [notes](./daily_notes/day_026.md) | [code](./05_vaes_gans/day_026_vae.py) | [blog](./blog_posts/day_026.md) |
| 27 | GANs — DCGAN Implementation | [0.5 VAEs & GANs](https://learn.arena.education/chapter0_fundamentals/05_vaes_gans/) | [notes](./daily_notes/day_027.md) | [code](./05_vaes_gans/day_027_gan.py) | [blog](./blog_posts/day_027.md) |
| 28 | Module 0 Capstone + Review | — | [notes](./daily_notes/day_028.md) | [code](./05_vaes_gans/day_028_capstone.py) | [blog](./blog_posts/day_028.md) |

</details>

---

### MODULE 1 — Mechanistic Interpretability
**ARENA Chapter 1 · Days 29–65**

<details>
<summary><strong>Transformers from Scratch (Days 29–33)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 29 | Attention Mechanism | [1.1 Transformers](https://learn.arena.education/chapter1_transformer_interp/01_transformers/) | [notes](./daily_notes/day_029.md) | [code](./06_transformers/day_029_attention.py) | [blog](./blog_posts/day_029.md) |
| 30 | Transformer Blocks | [1.1 Transformers](https://learn.arena.education/chapter1_transformer_interp/01_transformers/) | [notes](./daily_notes/day_030.md) | [code](./06_transformers/day_030_blocks.py) | [blog](./blog_posts/day_030.md) |
| 31 | Full GPT-2 Architecture | [1.1 Transformers](https://learn.arena.education/chapter1_transformer_interp/01_transformers/) | [notes](./daily_notes/day_031.md) | [code](./06_transformers/day_031_gpt2.py) | [blog](./blog_posts/day_031.md) |
| 32 | Load Pretrained Weights | [1.1 Transformers](https://learn.arena.education/chapter1_transformer_interp/01_transformers/) | [notes](./daily_notes/day_032.md) | [code](./06_transformers/day_032_pretrained.py) | [blog](./blog_posts/day_032.md) |
| 33 | Sampling Strategies | [1.1 Transformers](https://learn.arena.education/chapter1_transformer_interp/01_transformers/) | [notes](./daily_notes/day_033.md) | [code](./06_transformers/day_033_sampling.py) | [blog](./blog_posts/day_033.md) |

</details>

<details>
<summary><strong>Intro to Mechanistic Interpretability (Days 34–38)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 34 | TransformerLens Setup & Activations | [1.2 Intro to Mech Interp](https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/) | [notes](./daily_notes/day_034.md) | [code](./07_mech_interp/day_034_tl_setup.py) | [blog](./blog_posts/day_034.md) |
| 35 | Logit Lens | [1.2 Intro to Mech Interp](https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/) | [notes](./daily_notes/day_035.md) | [code](./07_mech_interp/day_035_logit_lens.py) | [blog](./blog_posts/day_035.md) |
| 36 | Hooks Deep Dive | [1.2 Intro to Mech Interp](https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/) | [notes](./daily_notes/day_036.md) | [code](./07_mech_interp/day_036_hooks.py) | [blog](./blog_posts/day_036.md) |
| 37 | Attention Head Analysis + Induction Heads | [1.2 Intro to Mech Interp](https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/) | [notes](./daily_notes/day_037.md) | [code](./07_mech_interp/day_037_induction.py) | [blog](./blog_posts/day_037.md) |
| 38 | MLP Neuron Analysis | [1.2 Intro to Mech Interp](https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/) | [notes](./daily_notes/day_038.md) | [code](./07_mech_interp/day_038_neurons.py) | [blog](./blog_posts/day_038.md) |

</details>

<details>
<summary><strong>Linear Probes (Days 39–41)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 39 | Probing Theory | [1.3.1 Linear Probes](https://learn.arena.education/chapter1_transformer_interp/11_probing/) | [notes](./daily_notes/day_039.md) | [code](./07_mech_interp/day_039_probes_theory.py) | [blog](./blog_posts/day_039.md) |
| 40 | Probe Implementation on Coup | [1.3.1 Linear Probes](https://learn.arena.education/chapter1_transformer_interp/11_probing/) | [notes](./daily_notes/day_040.md) | [code](./07_mech_interp/day_040_coup_probes.py) | [blog](./blog_posts/day_040.md) |
| 41 | Probe Analysis + Failure Modes | [1.3.1 Linear Probes](https://learn.arena.education/chapter1_transformer_interp/11_probing/) | [notes](./daily_notes/day_041.md) | [code](./07_mech_interp/day_041_probe_analysis.py) | [blog](./blog_posts/day_041.md) |

</details>

<details>
<summary><strong>Function Vectors & Model Steering (Days 42–44)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 42 | Activation Patching | [1.3.2 Function Vectors](https://learn.arena.education/chapter1_transformer_interp/12_function_vectors/) | [notes](./daily_notes/day_042.md) | [code](./07_mech_interp/day_042_act_patch.py) | [blog](./blog_posts/day_042.md) |
| 43 | Function Vectors | [1.3.2 Function Vectors](https://learn.arena.education/chapter1_transformer_interp/12_function_vectors/) | [notes](./daily_notes/day_043.md) | [code](./07_mech_interp/day_043_func_vectors.py) | [blog](./blog_posts/day_043.md) |
| 44 | nnsight & Activation Steering | [1.3.2 Function Vectors](https://learn.arena.education/chapter1_transformer_interp/12_function_vectors/) | [notes](./daily_notes/day_044.md) | [code](./07_mech_interp/day_044_steering.py) | [blog](./blog_posts/day_044.md) |

</details>

<details>
<summary><strong>Sparse Autoencoders (Days 45–48)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 45 | SAE Theory | [1.3.3 SAEs](https://learn.arena.education/chapter1_transformer_interp/13_saes/) | [notes](./daily_notes/day_045.md) | [code](./07_mech_interp/day_045_sae_theory.py) | [blog](./blog_posts/day_045.md) |
| 46 | SAE Training on GPT-2 | [1.3.3 SAEs](https://learn.arena.education/chapter1_transformer_interp/13_saes/) | [notes](./daily_notes/day_046.md) | [code](./07_mech_interp/day_046_sae_train.py) | [blog](./blog_posts/day_046.md) |
| 47 | Feature Interpretation | [1.3.3 SAEs](https://learn.arena.education/chapter1_transformer_interp/13_saes/) | [notes](./daily_notes/day_047.md) | [code](./07_mech_interp/day_047_features.py) | [blog](./blog_posts/day_047.md) |
| 48 | SAE as Cognitive Monitor | [1.3.3 SAEs](https://learn.arena.education/chapter1_transformer_interp/13_saes/) | [notes](./daily_notes/day_048.md) | [code](./07_mech_interp/day_048_monitor.py) | [blog](./blog_posts/day_048.md) |

</details>

<details>
<summary><strong>Activation Oracles (Days 49–51)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 49 | Activation Oracles Intro | [1.3.4 Activation Oracles](https://learn.arena.education/chapter1_transformer_interp/14_activation_oracles/) | [notes](./daily_notes/day_049.md) | [code](./07_mech_interp/day_049_oracles_intro.py) | [blog](./blog_posts/day_049.md) |
| 50 | Hidden Knowledge Probing | [1.3.4 Activation Oracles](https://learn.arena.education/chapter1_transformer_interp/14_activation_oracles/) | [notes](./daily_notes/day_050.md) | [code](./07_mech_interp/day_050_hidden_knowledge.py) | [blog](./blog_posts/day_050.md) |
| 51 | Oracle Analysis + Visualization | [1.3.4 Activation Oracles](https://learn.arena.education/chapter1_transformer_interp/14_activation_oracles/) | [notes](./daily_notes/day_051.md) | [code](./07_mech_interp/day_051_oracle_analysis.py) | [blog](./blog_posts/day_051.md) |

</details>

<details>
<summary><strong>Circuit Analysis — Indirect Object Identification (Days 52–55)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 52 | IOI Task Setup | [1.4.1 IOI](https://learn.arena.education/chapter1_transformer_interp/21_ioi/) | [notes](./daily_notes/day_052.md) | [code](./07_mech_interp/day_052_ioi_setup.py) | [blog](./blog_posts/day_052.md) |
| 53 | Path Patching | [1.4.1 IOI](https://learn.arena.education/chapter1_transformer_interp/21_ioi/) | [notes](./daily_notes/day_053.md) | [code](./07_mech_interp/day_053_path_patching.py) | [blog](./blog_posts/day_053.md) |
| 54 | Circuit Discovery | [1.4.1 IOI](https://learn.arena.education/chapter1_transformer_interp/21_ioi/) | [notes](./daily_notes/day_054.md) | [code](./07_mech_interp/day_054_circuit_map.py) | [blog](./blog_posts/day_054.md) |
| 55 | Circuit Validation via Ablation | [1.4.1 IOI](https://learn.arena.education/chapter1_transformer_interp/21_ioi/) | [notes](./daily_notes/day_055.md) | [code](./07_mech_interp/day_055_ablation.py) | [blog](./blog_posts/day_055.md) |

</details>

<details>
<summary><strong>SAE Circuits (Days 56–58)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 56 | SAE Circuit Theory | [1.4.2 SAE Circuits](https://learn.arena.education/chapter1_transformer_interp/22_sae_circuits/) | [notes](./daily_notes/day_056.md) | [code](./07_mech_interp/day_056_sae_circuits.py) | [blog](./blog_posts/day_056.md) |
| 57 | Feature Tracing Through Layers | [1.4.2 SAE Circuits](https://learn.arena.education/chapter1_transformer_interp/22_sae_circuits/) | [notes](./daily_notes/day_057.md) | [code](./07_mech_interp/day_057_feature_flow.py) | [blog](./blog_posts/day_057.md) |
| 58 | SAE + IOI Combined Analysis | [1.4.2 SAE Circuits](https://learn.arena.education/chapter1_transformer_interp/22_sae_circuits/) | [notes](./daily_notes/day_058.md) | [code](./07_mech_interp/day_058_sae_ioi.py) | [blog](./blog_posts/day_058.md) |

</details>

<details>
<summary><strong>Toy Models (Days 59–65)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 59 | Balanced Bracket Classifier | [1.5.1 Brackets](https://learn.arena.education/chapter1_transformer_interp/31_brackets/) | [notes](./daily_notes/day_059.md) | [code](./07_mech_interp/day_059_brackets.py) | [blog](./blog_posts/day_059.md) |
| 60 | Bracket Circuit Reverse-Engineering | [1.5.1 Brackets](https://learn.arena.education/chapter1_transformer_interp/31_brackets/) | [notes](./daily_notes/day_060.md) | [code](./07_mech_interp/day_060_brackets_circuit.py) | [blog](./blog_posts/day_060.md) |
| 61 | Grokking — Train + Observe | [1.5.2 Grokking](https://learn.arena.education/chapter1_transformer_interp/32_grokking/) | [notes](./daily_notes/day_061.md) | [code](./07_mech_interp/day_061_grokking.py) | [blog](./blog_posts/day_061.md) |
| 62 | Fourier Circuits in Grokking | [1.5.2 Grokking](https://learn.arena.education/chapter1_transformer_interp/32_grokking/) | [notes](./daily_notes/day_062.md) | [code](./07_mech_interp/day_062_fourier.py) | [blog](./blog_posts/day_062.md) |
| 63 | OthelloGPT World Model Analysis | [1.5.3 OthelloGPT](https://learn.arena.education/chapter1_transformer_interp/33_othellogpt/) | [notes](./daily_notes/day_063.md) | [code](./07_mech_interp/day_063_othello.py) | [blog](./blog_posts/day_063.md) |
| 64 | Superposition — Toy Model Replication | [1.5.4 Superposition](https://learn.arena.education/chapter1_transformer_interp/34_superposition/) | [notes](./daily_notes/day_064.md) | [code](./07_mech_interp/day_064_superposition.py) | [blog](./blog_posts/day_064.md) |
| 65 | SAE on Superposition + Module 1 Review | [1.5.4 Superposition](https://learn.arena.education/chapter1_transformer_interp/34_superposition/) | [notes](./daily_notes/day_065.md) | [code](./07_mech_interp/day_065_sae_superposition.py) | [blog](./blog_posts/day_065.md) |

</details>

---

### MODULE 2 — Reinforcement Learning
**ARENA Chapter 2 · Days 66–76**

<details>
<summary><strong>Intro to RL (Days 66–68)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 66 | MDPs, Bandits, Value Functions | [2.1 Intro to RL](https://learn.arena.education/chapter2_rl/01_intro_rl/) | [notes](./daily_notes/day_066.md) | [code](./08_rl/day_066_bandits.py) | [blog](./blog_posts/day_066.md) |
| 67 | Q-Learning & Tabular RL | [2.1 Intro to RL](https://learn.arena.education/chapter2_rl/01_intro_rl/) | [notes](./daily_notes/day_067.md) | [code](./08_rl/day_067_qlearning.py) | [blog](./blog_posts/day_067.md) |
| 68 | Policy & Value Iteration | [2.1 Intro to RL](https://learn.arena.education/chapter2_rl/01_intro_rl/) | [notes](./daily_notes/day_068.md) | [code](./08_rl/day_068_policy_iter.py) | [blog](./blog_posts/day_068.md) |

</details>

<details>
<summary><strong>DQN & VPG (Days 69–71)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 69 | Deep Q-Network (DQN) | [2.2 DQN & VPG](https://learn.arena.education/chapter2_rl/02_dqn_vpg/) | [notes](./daily_notes/day_069.md) | [code](./08_rl/day_069_dqn.py) | [blog](./blog_posts/day_069.md) |
| 70 | Vanilla Policy Gradient | [2.2 DQN & VPG](https://learn.arena.education/chapter2_rl/02_dqn_vpg/) | [notes](./daily_notes/day_070.md) | [code](./08_rl/day_070_vpg.py) | [blog](./blog_posts/day_070.md) |
| 71 | Actor-Critic (A2C) | [2.2 DQN & VPG](https://learn.arena.education/chapter2_rl/02_dqn_vpg/) | [notes](./daily_notes/day_071.md) | [code](./08_rl/day_071_a2c.py) | [blog](./blog_posts/day_071.md) |

</details>

<details>
<summary><strong>PPO (Days 72–74)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 72 | PPO Theory + Clipped Objective | [2.3 PPO](https://learn.arena.education/chapter2_rl/03_ppo/) | [notes](./daily_notes/day_072.md) | [code](./08_rl/day_072_ppo_theory.py) | [blog](./blog_posts/day_072.md) |
| 73 | PPO Implementation | [2.3 PPO](https://learn.arena.education/chapter2_rl/03_ppo/) | [notes](./daily_notes/day_073.md) | [code](./08_rl/day_073_ppo_impl.py) | [blog](./blog_posts/day_073.md) |
| 74 | PPO Debugging & Analysis | [2.3 PPO](https://learn.arena.education/chapter2_rl/03_ppo/) | [notes](./daily_notes/day_074.md) | [code](./08_rl/day_074_ppo_debug.py) | [blog](./blog_posts/day_074.md) |

</details>

<details>
<summary><strong>RLHF (Days 75–76)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 75 | RLHF Theory + Reward Model | [2.4 RLHF](https://learn.arena.education/chapter2_rl/04_rlhf/) | [notes](./daily_notes/day_075.md) | [code](./08_rl/day_075_reward_model.py) | [blog](./blog_posts/day_075.md) |
| 76 | Full RLHF Pipeline | [2.4 RLHF](https://learn.arena.education/chapter2_rl/04_rlhf/) | [notes](./daily_notes/day_076.md) | [code](./08_rl/day_076_rlhf_pipeline.py) | [blog](./blog_posts/day_076.md) |

</details>

---

### MODULE 3 — LLM Evaluations
**ARENA Chapter 3 · Days 77–86**

<details>
<summary><strong>Intro to Evals (Days 77–79)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 77 | Threat Models | [3.1 Intro to Evals](https://learn.arena.education/chapter3_llm_evals/01_intro_evals/) | [notes](./daily_notes/day_077.md) | [code](./09_evals/day_077_threat_model.md) | [blog](./blog_posts/day_077.md) |
| 78 | Eval Specification Design | [3.1 Intro to Evals](https://learn.arena.education/chapter3_llm_evals/01_intro_evals/) | [notes](./daily_notes/day_078.md) | [code](./09_evals/day_078_eval_spec.md) | [blog](./blog_posts/day_078.md) |
| 79 | Eval Pitfalls + Benchmark Gaming | [3.1 Intro to Evals](https://learn.arena.education/chapter3_llm_evals/01_intro_evals/) | [notes](./daily_notes/day_079.md) | [code](./09_evals/day_079_pitfalls.py) | [blog](./blog_posts/day_079.md) |

</details>

<details>
<summary><strong>Dataset Generation (Days 80–81)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 80 | LLM-Generated Eval Questions | [3.2 Dataset Generation](https://learn.arena.education/chapter3_llm_evals/02_dataset_gen/) | [notes](./daily_notes/day_080.md) | [code](./09_evals/day_080_gen_dataset.py) | [blog](./blog_posts/day_080.md) |
| 81 | Dataset Validation Pipeline | [3.2 Dataset Generation](https://learn.arena.education/chapter3_llm_evals/02_dataset_gen/) | [notes](./daily_notes/day_081.md) | [code](./09_evals/day_081_validate.py) | [blog](./blog_posts/day_081.md) |

</details>

<details>
<summary><strong>Running Evals with Inspect (Days 82–83)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 82 | Inspect Setup + First Eval | [3.3 Running Evals](https://learn.arena.education/chapter3_llm_evals/03_running_evals/) | [notes](./daily_notes/day_082.md) | [code](./09_evals/day_082_inspect_setup.py) | [blog](./blog_posts/day_082.md) |
| 83 | Custom Eval in Inspect | [3.3 Running Evals](https://learn.arena.education/chapter3_llm_evals/03_running_evals/) | [notes](./daily_notes/day_083.md) | [code](./09_evals/day_083_custom_eval.py) | [blog](./blog_posts/day_083.md) |

</details>

<details>
<summary><strong>LLM Agents (Days 84–86)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 84 | Agent Scaffolding (Tool Use) | [3.4 LLM Agents](https://learn.arena.education/chapter3_llm_evals/04_llm_agents/) | [notes](./daily_notes/day_084.md) | [code](./09_evals/day_084_agent.py) | [blog](./blog_posts/day_084.md) |
| 85 | Wikipedia Racing Agent | [3.4 LLM Agents](https://learn.arena.education/chapter3_llm_evals/04_llm_agents/) | [notes](./daily_notes/day_085.md) | [code](./09_evals/day_085_wiki_race.py) | [blog](./blog_posts/day_085.md) |
| 86 | Agent Eval Suite | [3.4 LLM Agents](https://learn.arena.education/chapter3_llm_evals/04_llm_agents/) | [notes](./daily_notes/day_086.md) | [code](./09_evals/day_086_agent_eval.py) | [blog](./blog_posts/day_086.md) |

</details>

---

### MODULE 4 — Alignment Science
**ARENA Chapter 4 · Days 87–100**

<details>
<summary><strong>Emergent Misalignment (Days 87–89)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 87 | Emergent Misalignment — Study | [4.1 Emergent Misalignment](https://learn.arena.education/chapter4_alignment_science/1_emergent_misalignment/) | [notes](./daily_notes/day_087.md) | [code](./10_alignment/day_087_em_study.py) | [blog](./blog_posts/day_087.md) |
| 88 | Misaligned Model Analysis | [4.1 Emergent Misalignment](https://learn.arena.education/chapter4_alignment_science/1_emergent_misalignment/) | [notes](./daily_notes/day_088.md) | [code](./10_alignment/day_088_em_analysis.py) | [blog](./blog_posts/day_088.md) |
| 89 | Mitigation Experiments | [4.1 Emergent Misalignment](https://learn.arena.education/chapter4_alignment_science/1_emergent_misalignment/) | [notes](./daily_notes/day_089.md) | [code](./10_alignment/day_089_mitigation.py) | [blog](./blog_posts/day_089.md) |

</details>

<details>
<summary><strong>Science of Misalignment (Days 90–91)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 90 | Black-Box Investigation Case Study 1 | [4.2 Science of Misalignment](https://learn.arena.education/chapter4_alignment_science/2_science_misalignment/) | [notes](./daily_notes/day_090.md) | [code](./10_alignment/day_090_blackbox_1.py) | [blog](./blog_posts/day_090.md) |
| 91 | Black-Box Investigation Case Study 2 | [4.2 Science of Misalignment](https://learn.arena.education/chapter4_alignment_science/2_science_misalignment/) | [notes](./daily_notes/day_091.md) | [code](./10_alignment/day_091_blackbox_2.py) | [blog](./blog_posts/day_091.md) |

</details>

<details>
<summary><strong>Interpreting Reasoning Models (Days 92–93)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 92 | Chain-of-Thought Faithfulness | [4.3 Reasoning Models](https://learn.arena.education/chapter4_alignment_science/3_reasoning_models/) | [notes](./daily_notes/day_092.md) | [code](./10_alignment/day_092_cot_faith.py) | [blog](./blog_posts/day_092.md) |
| 93 | Reasoning Model Interpretability | [4.3 Reasoning Models](https://learn.arena.education/chapter4_alignment_science/3_reasoning_models/) | [notes](./daily_notes/day_093.md) | [code](./10_alignment/day_093_reasoning_interp.py) | [blog](./blog_posts/day_093.md) |

</details>

<details>
<summary><strong>LLM Psychology & Persona Vectors (Days 94–95)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 94 | Persona Vector Extraction | [4.4 Persona Vectors](https://learn.arena.education/chapter4_alignment_science/4_persona_vectors/) | [notes](./daily_notes/day_094.md) | [code](./10_alignment/day_094_persona.py) | [blog](./blog_posts/day_094.md) |
| 95 | Psychological Consistency Probing | [4.4 Persona Vectors](https://learn.arena.education/chapter4_alignment_science/4_persona_vectors/) | [notes](./daily_notes/day_095.md) | [code](./10_alignment/day_095_psych_probe.py) | [blog](./blog_posts/day_095.md) |

</details>

<details>
<summary><strong>Investigator Agents (Days 96–98)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 96 | Investigator Agent Setup | [4.5 Investigator Agents](https://learn.arena.education/chapter4_alignment_science/5_investigator_agents/) | [notes](./daily_notes/day_096.md) | [code](./10_alignment/day_096_inv_agent.py) | [blog](./blog_posts/day_096.md) |
| 97 | Automated Investigation Experiments | [4.5 Investigator Agents](https://learn.arena.education/chapter4_alignment_science/5_investigator_agents/) | [notes](./daily_notes/day_097.md) | [code](./10_alignment/day_097_inv_experiments.py) | [blog](./blog_posts/day_097.md) |
| 98 | Investigator Agent Evaluation | [4.5 Investigator Agents](https://learn.arena.education/chapter4_alignment_science/5_investigator_agents/) | [notes](./daily_notes/day_098.md) | [code](./10_alignment/day_098_inv_eval.py) | [blog](./blog_posts/day_098.md) |

</details>

<details>
<summary><strong>Final Review & Capstone (Days 99–100)</strong></summary>

| Day | Topic | Notes | Blog |
|---|---|---|---|
| 99 | Full Curriculum Review + Research Gaps | [notes](./daily_notes/day_099.md) | [blog](./blog_posts/day_099.md) |
| 100 | Capstone: Research Proposal | [notes](./daily_notes/day_100.md) | [blog](./blog_posts/day_100.md) |

</details>

---

## Repository Structure

```
arena-mechanistic-learning/
│
├── 00_prerequisites/          # Days 1–5   · PyTorch, einops, linear algebra
├── 01_ray_tracing/            # Days 6–10  · Batched ops, 3D rendering
├── 02_cnns_resnets/           # Days 11–14 · Convolutions, ResNet, CIFAR-10
├── 03_optimization/           # Days 15–19 · SGD, Adam, W&B, sweeps
├── 04_backprop/               # Days 20–24 · Autograd engine from scratch
├── 05_vaes_gans/              # Days 25–28 · VAEs, GANs, generative models
│
├── 06_transformers/           # Days 29–33 · GPT-2 from scratch, sampling
├── 07_mech_interp/            # Days 34–65 · TransformerLens, probes, SAEs,
│                              #              circuits, IOI, grokking, OthelloGPT
│
├── 08_rl/                     # Days 66–76 · Bandits, Q-learning, DQN, PPO, RLHF
│
├── 09_evals/                  # Days 77–86 · Threat models, Inspect, LLM agents
│
├── 10_alignment/              # Days 87–100 · Emergent misalignment, reasoning
│                              #               models, persona vectors, inv. agents
│
├── blog_posts/                # Daily blog posts (markdown)
│   ├── day_001.md
│   └── ...
│
└── daily_notes/               # Short daily notes and reflections
    ├── day_001.md
    └── ...
```

---

## Motivation

Modern ML frameworks automate gradient computation. While powerful, that abstraction hides the underlying mechanism of learning.

At its core, learning in neural networks is:

> Routing responsibility signals (gradients) backward through a computational graph.

By deriving and implementing core algorithms directly — and then using interpretability tools to look inside trained models — I aim to understand:

- How gradients flow (and fail)
- How representations form and compress into features
- How policies learn over time in RL
- How these mechanisms relate to alignment and safety
- How to detect when a model's internals don't match its outputs

---

## Attribution & Credit

This repository documents my independent implementations and notes while studying the ARENA curriculum.

Full credit goes to the ARENA team for creating this program.

**Official ARENA resources:**
- Curriculum: https://learn.arena.education/
- Website: https://www.arena.education/

All original course materials belong to ARENA. This repository contains only my personal implementations, derivations, experiments, and blog posts.

**This is not an official ARENA resource.**

---

## Where I Write

Longer blog posts and weekly summaries get posted to:
- [Personal](https://kundan-kumarr.github.io/blog/)
- [SubStack](https://neuravp.substack.com/)

---


