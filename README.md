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
| 0 | 1–14 | Fundamentals (PyTorch → VAEs & GANs) | Started |
| 1 | 15–29 | Reinforcement Learning + RLHF | Not started |
| 2 | 30–70 | Mechanistic Interpretability | Not started |
| 3 | 71–82 | LLM Evaluations | Not started |
| 4 | 83–100 | Alignment Science | Not started |

---

## Daily Log

### MODULE 0 — Fundamentals
**ARENA Chapter 0 · Days 1–14**

<details>
<summary><strong>Prerequisites & PyTorch Fluency (Days 1–2)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 1 | PyTorch, einops, einsum | [0.0 Prerequisites](https://learn.arena.education/chapter0_fundamentals/00_prereqs/) | [notes](./daily_notes/day_001.md) | [code](./01_backprop/day_001_tensors.py) | [blog](./blog_posts/day_001.md) |
| 2 | Tensor Manipulation + Linear Algebra in Code | [0.0 Prerequisites](https://learn.arena.education/chapter0_fundamentals/00_prereqs/) | [notes](./daily_notes/day_002.md) | [code](./01_backprop/day_002_linalg.py) | [blog](./blog_posts/day_002.md) |

</details>

<details>
<summary><strong>Ray Tracing & Batched Operations (Days 3–5)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 3 | Ray Tracing Intro + Batched Rays | [0.1 Ray Tracing](https://learn.arena.education/chapter0_fundamentals/01_ray_tracing/) | [notes](./daily_notes/day_003.md) | [code](./01_ray_tracing/day_003_rays.py) | [blog](./blog_posts/day_003.md) |
| 4 | Triangle Meshes + Full Render Pipeline | [0.1 Ray Tracing](https://learn.arena.education/chapter0_fundamentals/01_ray_tracing/) | [notes](./daily_notes/day_004.md) | [code](./01_ray_tracing/day_004_render.py) | [blog](./blog_posts/day_004.md) |
| 5 | Performance Optimization | [0.1 Ray Tracing](https://learn.arena.education/chapter0_fundamentals/01_ray_tracing/) | [notes](./daily_notes/day_005.md) | [code](./01_ray_tracing/day_005_perf.py) | [blog](./blog_posts/day_005.md) |

</details>

<details>
<summary><strong>CNNs & ResNets (Days 6–8)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 6 | Conv Operations from Scratch | [0.2 CNNs & ResNets](https://learn.arena.education/chapter0_fundamentals/02_cnns/) | [notes](./daily_notes/day_006.md) | [code](./02_cnns_resnets/day_006_conv.py) | [blog](./blog_posts/day_006.md) |
| 7 | MNIST Classifier | [0.2 CNNs & ResNets](https://learn.arena.education/chapter0_fundamentals/02_cnns/) | [notes](./daily_notes/day_007.md) | [code](./02_cnns_resnets/day_007_mnist.py) | [blog](./blog_posts/day_007.md) |
| 8 | ResNets — Architecture + CIFAR-10 Training | [0.2 CNNs & ResNets](https://learn.arena.education/chapter0_fundamentals/02_cnns/) | [notes](./daily_notes/day_008.md) | [code](./02_cnns_resnets/day_008_resnet.py) | [blog](./blog_posts/day_008.md) |

</details>

<details>
<summary><strong>Optimization (Days 9–10)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 9 | SGD + Adam from Scratch | [0.3 Optimization](https://learn.arena.education/chapter0_fundamentals/03_optimization/) | [notes](./daily_notes/day_009.md) | [code](./03_optimization/day_009_optimizers.py) | [blog](./blog_posts/day_009.md) |
| 10 | Weights & Biases + Hyperparameter Sweeps | [0.3 Optimization](https://learn.arena.education/chapter0_fundamentals/03_optimization/) | [notes](./daily_notes/day_010.md) | [code](./03_optimization/day_010_wandb.py) | [blog](./blog_posts/day_010.md) |

</details>

<details>
<summary><strong>Backpropagation from Scratch (Days 11–12)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 11 | Autograd Engine + Chain Rule | [0.4 Backpropagation](https://learn.arena.education/chapter0_fundamentals/04_backprop/) | [notes](./daily_notes/day_011.md) | [code](./04_backprop/day_011_autograd.py) | [blog](./blog_posts/day_011.md) |
| 12 | Train MLP with Custom Autograd + Gradient Checking | [0.4 Backpropagation](https://learn.arena.education/chapter0_fundamentals/04_backprop/) | [notes](./daily_notes/day_012.md) | [code](./04_backprop/day_012_mlp_gradcheck.py) | [blog](./blog_posts/day_012.md) |

</details>

<details>
<summary><strong>VAEs & GANs (Days 13–14)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 13 | VAEs — Theory + ELBO Training | [0.5 VAEs & GANs](https://learn.arena.education/chapter0_fundamentals/05_vaes_gans/) | [notes](./daily_notes/day_013.md) | [code](./05_vaes_gans/day_013_vae.py) | [blog](./blog_posts/day_013.md) |
| 14 | GANs — DCGAN Implementation | [0.5 VAEs & GANs](https://learn.arena.education/chapter0_fundamentals/05_vaes_gans/) | [notes](./daily_notes/day_014.md) | [code](./05_vaes_gans/day_014_gan.py) | [blog](./blog_posts/day_014.md) |

</details>

---

### MODULE 2 — Reinforcement Learning
**ARENA Chapter 2 · Days 15–29**

<details>
<summary><strong>Intro to RL (Days 15–17)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 15 | MDPs, Bandits, Value Functions | [2.1 Intro to RL](https://learn.arena.education/chapter2_rl/01_intro_rl/) | [notes](./daily_notes/day_015.md) | [code](./06_rl/day_015_bandits.py) | [blog](./blog_posts/day_015.md) |
| 16 | Q-Learning & Tabular RL | [2.1 Intro to RL](https://learn.arena.education/chapter2_rl/01_intro_rl/) | [notes](./daily_notes/day_016.md) | [code](./06_rl/day_016_qlearning.py) | [blog](./blog_posts/day_016.md) |
| 17 | Policy & Value Iteration | [2.1 Intro to RL](https://learn.arena.education/chapter2_rl/01_intro_rl/) | [notes](./daily_notes/day_017.md) | [code](./06_rl/day_017_policy_iter.py) | [blog](./blog_posts/day_017.md) |

</details>

<details>
<summary><strong>DQN & VPG (Days 18–20)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 18 | Deep Q-Network (DQN) | [2.2 DQN & VPG](https://learn.arena.education/chapter2_rl/02_dqn_vpg/) | [notes](./daily_notes/day_018.md) | [code](./06_rl/day_018_dqn.py) | [blog](./blog_posts/day_018.md) |
| 19 | Vanilla Policy Gradient | [2.2 DQN & VPG](https://learn.arena.education/chapter2_rl/02_dqn_vpg/) | [notes](./daily_notes/day_019.md) | [code](./06_rl/day_019_vpg.py) | [blog](./blog_posts/day_019.md) |
| 20 | Actor-Critic (A2C) | [2.2 DQN & VPG](https://learn.arena.education/chapter2_rl/02_dqn_vpg/) | [notes](./daily_notes/day_020.md) | [code](./06_rl/day_020_a2c.py) | [blog](./blog_posts/day_020.md) |

</details>

<details>
<summary><strong>PPO (Days 21–23)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 21 | PPO Theory + Clipped Objective | [2.3 PPO](https://learn.arena.education/chapter2_rl/03_ppo/) | [notes](./daily_notes/day_021.md) | [code](./06_rl/day_021_ppo_theory.py) | [blog](./blog_posts/day_021.md) |
| 22 | PPO Implementation | [2.3 PPO](https://learn.arena.education/chapter2_rl/03_ppo/) | [notes](./daily_notes/day_022.md) | [code](./06_rl/day_022_ppo_impl.py) | [blog](./blog_posts/day_022.md) |
| 23 | PPO Debugging & Analysis | [2.3 PPO](https://learn.arena.education/chapter2_rl/03_ppo/) | [notes](./daily_notes/day_023.md) | [code](./06_rl/day_023_ppo_debug.py) | [blog](./blog_posts/day_023.md) |

</details>

<details>
<summary><strong>RLHF (Days 24–29)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 24 | RLHF Theory + Reward Model | [2.4 RLHF](https://learn.arena.education/chapter2_rl/04_rlhf/) | [notes](./daily_notes/day_024.md) | [code](./06_rl/day_024_reward_model.py) | [blog](./blog_posts/day_024.md) |
| 25 | Full RLHF Pipeline | [2.4 RLHF](https://learn.arena.education/chapter2_rl/04_rlhf/) | [notes](./daily_notes/day_025.md) | [code](./06_rl/day_025_rlhf_pipeline.py) | [blog](./blog_posts/day_025.md) |
| 26 | RLHF — Interpretability Lens | [2.4 RLHF](https://learn.arena.education/chapter2_rl/04_rlhf/) | [notes](./daily_notes/day_026.md) | [code](./06_rl/day_026_rlhf_interp.py) | [blog](./blog_posts/day_026.md) |
| 27 | Reward Hacking | — | [notes](./daily_notes/day_027.md) | [code](./06_rl/day_027_reward_hacking.py) | [blog](./blog_posts/day_027.md) |
| 28 | Specification Gaming | — | [notes](./daily_notes/day_028.md) | [code](./06_rl/day_028_spec_gaming.py) | [blog](./blog_posts/day_028.md) |
| 29 | Module 2 Review | — | [notes](./daily_notes/day_029.md) | [code](./06_rl/day_029_review.py) | [blog](./blog_posts/day_029.md) |

</details>

---

### MODULE 1 — Mechanistic Interpretability
**ARENA Chapter 1 · Days 30–70**

<details>
<summary><strong>Transformers from Scratch (Days 30–34)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 30 | Attention Mechanism | [1.1 Transformers](https://learn.arena.education/chapter1_transformer_interp/01_transformers/) | [notes](./daily_notes/day_030.md) | [code](./07_transformers/day_030_attention.py) | [blog](./blog_posts/day_030.md) |
| 31 | Transformer Blocks | [1.1 Transformers](https://learn.arena.education/chapter1_transformer_interp/01_transformers/) | [notes](./daily_notes/day_031.md) | [code](./07_transformers/day_031_blocks.py) | [blog](./blog_posts/day_031.md) |
| 32 | Full GPT-2 Architecture | [1.1 Transformers](https://learn.arena.education/chapter1_transformer_interp/01_transformers/) | [notes](./daily_notes/day_032.md) | [code](./07_transformers/day_032_gpt2.py) | [blog](./blog_posts/day_032.md) |
| 33 | Load Pretrained Weights | [1.1 Transformers](https://learn.arena.education/chapter1_transformer_interp/01_transformers/) | [notes](./daily_notes/day_033.md) | [code](./07_transformers/day_033_pretrained.py) | [blog](./blog_posts/day_033.md) |
| 34 | Sampling Strategies | [1.1 Transformers](https://learn.arena.education/chapter1_transformer_interp/01_transformers/) | [notes](./daily_notes/day_034.md) | [code](./07_transformers/day_034_sampling.py) | [blog](./blog_posts/day_034.md) |

</details>

<details>
<summary><strong>Intro to Mechanistic Interpretability (Days 35–39)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 35 | TransformerLens Setup & Activations | [1.2 Intro to Mech Interp](https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/) | [notes](./daily_notes/day_035.md) | [code](./08_mech_interp/day_035_tl_setup.py) | [blog](./blog_posts/day_035.md) |
| 36 | Logit Lens | [1.2 Intro to Mech Interp](https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/) | [notes](./daily_notes/day_036.md) | [code](./08_mech_interp/day_036_logit_lens.py) | [blog](./blog_posts/day_036.md) |
| 37 | Hooks Deep Dive | [1.2 Intro to Mech Interp](https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/) | [notes](./daily_notes/day_037.md) | [code](./08_mech_interp/day_037_hooks.py) | [blog](./blog_posts/day_037.md) |
| 38 | Attention Head Analysis + Induction Heads | [1.2 Intro to Mech Interp](https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/) | [notes](./daily_notes/day_038.md) | [code](./08_mech_interp/day_038_induction.py) | [blog](./blog_posts/day_038.md) |
| 39 | MLP Neuron Analysis | [1.2 Intro to Mech Interp](https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/) | [notes](./daily_notes/day_039.md) | [code](./08_mech_interp/day_039_neurons.py) | [blog](./blog_posts/day_039.md) |

</details>

<details>
<summary><strong>Linear Probes (Days 40–42)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 40 | Probing Theory | [1.3.1 Linear Probes](https://learn.arena.education/chapter1_transformer_interp/11_probing/) | [notes](./daily_notes/day_040.md) | [code](./08_mech_interp/day_040_probes_theory.py) | [blog](./blog_posts/day_040.md) |
| 41 | Probe Implementation on Coup | [1.3.1 Linear Probes](https://learn.arena.education/chapter1_transformer_interp/11_probing/) | [notes](./daily_notes/day_041.md) | [code](./08_mech_interp/day_041_coup_probes.py) | [blog](./blog_posts/day_041.md) |
| 42 | Probe Analysis + Failure Modes | [1.3.1 Linear Probes](https://learn.arena.education/chapter1_transformer_interp/11_probing/) | [notes](./daily_notes/day_042.md) | [code](./08_mech_interp/day_042_probe_analysis.py) | [blog](./blog_posts/day_042.md) |

</details>

<details>
<summary><strong>Function Vectors & Model Steering (Days 43–45)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 43 | Activation Patching | [1.3.2 Function Vectors](https://learn.arena.education/chapter1_transformer_interp/12_function_vectors/) | [notes](./daily_notes/day_043.md) | [code](./08_mech_interp/day_043_act_patch.py) | [blog](./blog_posts/day_043.md) |
| 44 | Function Vectors | [1.3.2 Function Vectors](https://learn.arena.education/chapter1_transformer_interp/12_function_vectors/) | [notes](./daily_notes/day_044.md) | [code](./08_mech_interp/day_044_func_vectors.py) | [blog](./blog_posts/day_044.md) |
| 45 | nnsight & Activation Steering | [1.3.2 Function Vectors](https://learn.arena.education/chapter1_transformer_interp/12_function_vectors/) | [notes](./daily_notes/day_045.md) | [code](./08_mech_interp/day_045_steering.py) | [blog](./blog_posts/day_045.md) |

</details>

<details>
<summary><strong>Sparse Autoencoders (Days 46–49)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 46 | SAE Theory | [1.3.3 SAEs](https://learn.arena.education/chapter1_transformer_interp/13_saes/) | [notes](./daily_notes/day_046.md) | [code](./08_mech_interp/day_046_sae_theory.py) | [blog](./blog_posts/day_046.md) |
| 47 | SAE Training on GPT-2 | [1.3.3 SAEs](https://learn.arena.education/chapter1_transformer_interp/13_saes/) | [notes](./daily_notes/day_047.md) | [code](./08_mech_interp/day_047_sae_train.py) | [blog](./blog_posts/day_047.md) |
| 48 | Feature Interpretation | [1.3.3 SAEs](https://learn.arena.education/chapter1_transformer_interp/13_saes/) | [notes](./daily_notes/day_048.md) | [code](./08_mech_interp/day_048_features.py) | [blog](./blog_posts/day_048.md) |
| 49 | SAE as Cognitive Monitor | [1.3.3 SAEs](https://learn.arena.education/chapter1_transformer_interp/13_saes/) | [notes](./daily_notes/day_049.md) | [code](./08_mech_interp/day_049_monitor.py) | [blog](./blog_posts/day_049.md) |

</details>

<details>
<summary><strong>Activation Oracles (Days 50–52)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 50 | Activation Oracles Intro | [1.3.4 Activation Oracles](https://learn.arena.education/chapter1_transformer_interp/14_activation_oracles/) | [notes](./daily_notes/day_050.md) | [code](./08_mech_interp/day_050_oracles_intro.py) | [blog](./blog_posts/day_050.md) |
| 51 | Hidden Knowledge Probing | [1.3.4 Activation Oracles](https://learn.arena.education/chapter1_transformer_interp/14_activation_oracles/) | [notes](./daily_notes/day_051.md) | [code](./08_mech_interp/day_051_hidden_knowledge.py) | [blog](./blog_posts/day_051.md) |
| 52 | Oracle Analysis + Visualization | [1.3.4 Activation Oracles](https://learn.arena.education/chapter1_transformer_interp/14_activation_oracles/) | [notes](./daily_notes/day_052.md) | [code](./08_mech_interp/day_052_oracle_analysis.py) | [blog](./blog_posts/day_052.md) |

</details>

<details>
<summary><strong>Circuit Analysis — Indirect Object Identification (Days 53–56)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 53 | IOI Task Setup | [1.4.1 IOI](https://learn.arena.education/chapter1_transformer_interp/21_ioi/) | [notes](./daily_notes/day_053.md) | [code](./08_mech_interp/day_053_ioi_setup.py) | [blog](./blog_posts/day_053.md) |
| 54 | Path Patching | [1.4.1 IOI](https://learn.arena.education/chapter1_transformer_interp/21_ioi/) | [notes](./daily_notes/day_054.md) | [code](./08_mech_interp/day_054_path_patching.py) | [blog](./blog_posts/day_054.md) |
| 55 | Circuit Discovery | [1.4.1 IOI](https://learn.arena.education/chapter1_transformer_interp/21_ioi/) | [notes](./daily_notes/day_055.md) | [code](./08_mech_interp/day_055_circuit_map.py) | [blog](./blog_posts/day_055.md) |
| 56 | Circuit Validation via Ablation | [1.4.1 IOI](https://learn.arena.education/chapter1_transformer_interp/21_ioi/) | [notes](./daily_notes/day_056.md) | [code](./08_mech_interp/day_056_ablation.py) | [blog](./blog_posts/day_056.md) |

</details>

<details>
<summary><strong>SAE Circuits (Days 57–59)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 57 | SAE Circuit Theory | [1.4.2 SAE Circuits](https://learn.arena.education/chapter1_transformer_interp/22_sae_circuits/) | [notes](./daily_notes/day_057.md) | [code](./08_mech_interp/day_057_sae_circuits.py) | [blog](./blog_posts/day_057.md) |
| 58 | Feature Tracing Through Layers | [1.4.2 SAE Circuits](https://learn.arena.education/chapter1_transformer_interp/22_sae_circuits/) | [notes](./daily_notes/day_058.md) | [code](./08_mech_interp/day_058_feature_flow.py) | [blog](./blog_posts/day_058.md) |
| 59 | SAE + IOI Combined Analysis | [1.4.2 SAE Circuits](https://learn.arena.education/chapter1_transformer_interp/22_sae_circuits/) | [notes](./daily_notes/day_059.md) | [code](./08_mech_interp/day_059_sae_ioi.py) | [blog](./blog_posts/day_059.md) |

</details>

<details>
<summary><strong>Toy Models (Days 60–70)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 60 | Balanced Bracket Classifier | [1.5.1 Brackets](https://learn.arena.education/chapter1_transformer_interp/31_brackets/) | [notes](./daily_notes/day_060.md) | [code](./08_mech_interp/day_060_brackets.py) | [blog](./blog_posts/day_060.md) |
| 61 | Bracket Circuit Reverse-Engineering | [1.5.1 Brackets](https://learn.arena.education/chapter1_transformer_interp/31_brackets/) | [notes](./daily_notes/day_061.md) | [code](./08_mech_interp/day_061_brackets_circuit.py) | [blog](./blog_posts/day_061.md) |
| 62 | Grokking — Train + Observe | [1.5.2 Grokking](https://learn.arena.education/chapter1_transformer_interp/32_grokking/) | [notes](./daily_notes/day_062.md) | [code](./08_mech_interp/day_062_grokking.py) | [blog](./blog_posts/day_062.md) |
| 63 | Fourier Circuits in Grokking | [1.5.2 Grokking](https://learn.arena.education/chapter1_transformer_interp/32_grokking/) | [notes](./daily_notes/day_063.md) | [code](./08_mech_interp/day_063_fourier.py) | [blog](./blog_posts/day_063.md) |
| 64 | Grokking — Progress Measures | [1.5.2 Grokking](https://learn.arena.education/chapter1_transformer_interp/32_grokking/) | [notes](./daily_notes/day_064.md) | [code](./08_mech_interp/day_064_progress.py) | [blog](./blog_posts/day_064.md) |
| 65 | OthelloGPT World Model Analysis | [1.5.3 OthelloGPT](https://learn.arena.education/chapter1_transformer_interp/33_othellogpt/) | [notes](./daily_notes/day_065.md) | [code](./08_mech_interp/day_065_othello.py) | [blog](./blog_posts/day_065.md) |
| 66 | OthelloGPT — Interventions | [1.5.3 OthelloGPT](https://learn.arena.education/chapter1_transformer_interp/33_othellogpt/) | [notes](./daily_notes/day_066.md) | [code](./08_mech_interp/day_066_othello_interv.py) | [blog](./blog_posts/day_066.md) |
| 67 | Superposition — Toy Model Replication | [1.5.4 Superposition](https://learn.arena.education/chapter1_transformer_interp/34_superposition/) | [notes](./daily_notes/day_067.md) | [code](./08_mech_interp/day_067_superposition.py) | [blog](./blog_posts/day_067.md) |
| 68 | Superposition — Feature Geometry | [1.5.4 Superposition](https://learn.arena.education/chapter1_transformer_interp/34_superposition/) | [notes](./daily_notes/day_068.md) | [code](./08_mech_interp/day_068_geometry.py) | [blog](./blog_posts/day_068.md) |
| 69 | SAE on Superposition Model | [1.5.4 Superposition](https://learn.arena.education/chapter1_transformer_interp/34_superposition/) | [notes](./daily_notes/day_069.md) | [code](./08_mech_interp/day_069_sae_superposition.py) | [blog](./blog_posts/day_069.md) |
| 70 | Module 1 Review | — | [notes](./daily_notes/day_070.md) | [code](./08_mech_interp/day_070_review.py) | [blog](./blog_posts/day_070.md) |

</details>

---

### MODULE 3 — LLM Evaluations
**ARENA Chapter 3 · Days 71–82**

<details>
<summary><strong>Intro to Evals (Days 71–73)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 71 | Threat Models | [3.1 Intro to Evals](https://learn.arena.education/chapter3_llm_evals/01_intro_evals/) | [notes](./daily_notes/day_071.md) | [code](./09_evals/day_071_threat_model.md) | [blog](./blog_posts/day_071.md) |
| 72 | Eval Specification Design | [3.1 Intro to Evals](https://learn.arena.education/chapter3_llm_evals/01_intro_evals/) | [notes](./daily_notes/day_072.md) | [code](./09_evals/day_072_eval_spec.md) | [blog](./blog_posts/day_072.md) |
| 73 | Eval Pitfalls + Benchmark Gaming | [3.1 Intro to Evals](https://learn.arena.education/chapter3_llm_evals/01_intro_evals/) | [notes](./daily_notes/day_073.md) | [code](./09_evals/day_073_pitfalls.py) | [blog](./blog_posts/day_073.md) |

</details>

<details>
<summary><strong>Dataset Generation (Days 74–75)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 74 | LLM-Generated Eval Questions | [3.2 Dataset Generation](https://learn.arena.education/chapter3_llm_evals/02_dataset_gen/) | [notes](./daily_notes/day_074.md) | [code](./09_evals/day_074_gen_dataset.py) | [blog](./blog_posts/day_074.md) |
| 75 | Dataset Validation Pipeline | [3.2 Dataset Generation](https://learn.arena.education/chapter3_llm_evals/02_dataset_gen/) | [notes](./daily_notes/day_075.md) | [code](./09_evals/day_075_validate.py) | [blog](./blog_posts/day_075.md) |

</details>

<details>
<summary><strong>Running Evals with Inspect (Days 76–77)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 76 | Inspect Setup + First Eval | [3.3 Running Evals](https://learn.arena.education/chapter3_llm_evals/03_running_evals/) | [notes](./daily_notes/day_076.md) | [code](./09_evals/day_076_inspect_setup.py) | [blog](./blog_posts/day_076.md) |
| 77 | Custom Eval in Inspect | [3.3 Running Evals](https://learn.arena.education/chapter3_llm_evals/03_running_evals/) | [notes](./daily_notes/day_077.md) | [code](./09_evals/day_077_custom_eval.py) | [blog](./blog_posts/day_077.md) |

</details>

<details>
<summary><strong>LLM Agents (Days 78–82)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 78 | Agent Scaffolding (Tool Use) | [3.4 LLM Agents](https://learn.arena.education/chapter3_llm_evals/04_llm_agents/) | [notes](./daily_notes/day_078.md) | [code](./09_evals/day_078_agent.py) | [blog](./blog_posts/day_078.md) |
| 79 | Wikipedia Racing Agent | [3.4 LLM Agents](https://learn.arena.education/chapter3_llm_evals/04_llm_agents/) | [notes](./daily_notes/day_079.md) | [code](./09_evals/day_079_wiki_race.py) | [blog](./blog_posts/day_079.md) |
| 80 | Agent Eval Suite | [3.4 LLM Agents](https://learn.arena.education/chapter3_llm_evals/04_llm_agents/) | [notes](./daily_notes/day_080.md) | [code](./09_evals/day_080_agent_eval.py) | [blog](./blog_posts/day_080.md) |
| 81 | Evals + Interpretability Crossover | — | [notes](./daily_notes/day_081.md) | [code](./09_evals/day_081_interp_evals.py) | [blog](./blog_posts/day_081.md) |
| 82 | Module 3 Review + Eval Report | — | [notes](./daily_notes/day_082.md) | [code](./09_evals/day_082_review.py) | [blog](./blog_posts/day_082.md) |

</details>

---

### MODULE 4 — Alignment Science
**ARENA Chapter 4 · Days 83–100**

<details>
<summary><strong>Emergent Misalignment (Days 83–85)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 83 | Emergent Misalignment — Study | [4.1 Emergent Misalignment](https://learn.arena.education/chapter4_alignment_science/1_emergent_misalignment/) | [notes](./daily_notes/day_083.md) | [code](./10_alignment/day_083_em_study.py) | [blog](./blog_posts/day_083.md) |
| 84 | Misaligned Model Analysis | [4.1 Emergent Misalignment](https://learn.arena.education/chapter4_alignment_science/1_emergent_misalignment/) | [notes](./daily_notes/day_084.md) | [code](./10_alignment/day_084_em_analysis.py) | [blog](./blog_posts/day_084.md) |
| 85 | Mitigation Experiments | [4.1 Emergent Misalignment](https://learn.arena.education/chapter4_alignment_science/1_emergent_misalignment/) | [notes](./daily_notes/day_085.md) | [code](./10_alignment/day_085_mitigation.py) | [blog](./blog_posts/day_085.md) |

</details>

<details>
<summary><strong>Science of Misalignment (Days 86–87)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 86 | Black-Box Investigation Case Study 1 | [4.2 Science of Misalignment](https://learn.arena.education/chapter4_alignment_science/2_science_misalignment/) | [notes](./daily_notes/day_086.md) | [code](./10_alignment/day_086_blackbox_1.py) | [blog](./blog_posts/day_086.md) |
| 87 | Black-Box Investigation Case Study 2 | [4.2 Science of Misalignment](https://learn.arena.education/chapter4_alignment_science/2_science_misalignment/) | [notes](./daily_notes/day_087.md) | [code](./10_alignment/day_087_blackbox_2.py) | [blog](./blog_posts/day_087.md) |

</details>

<details>
<summary><strong>Interpreting Reasoning Models (Days 88–89)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 88 | Chain-of-Thought Faithfulness | [4.3 Reasoning Models](https://learn.arena.education/chapter4_alignment_science/3_reasoning_models/) | [notes](./daily_notes/day_088.md) | [code](./10_alignment/day_088_cot_faith.py) | [blog](./blog_posts/day_088.md) |
| 89 | Reasoning Model Interpretability | [4.3 Reasoning Models](https://learn.arena.education/chapter4_alignment_science/3_reasoning_models/) | [notes](./daily_notes/day_089.md) | [code](./10_alignment/day_089_reasoning_interp.py) | [blog](./blog_posts/day_089.md) |

</details>

<details>
<summary><strong>LLM Psychology & Persona Vectors (Days 90–91)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 90 | Persona Vector Extraction | [4.4 Persona Vectors](https://learn.arena.education/chapter4_alignment_science/4_persona_vectors/) | [notes](./daily_notes/day_090.md) | [code](./10_alignment/day_090_persona.py) | [blog](./blog_posts/day_090.md) |
| 91 | Psychological Consistency Probing | [4.4 Persona Vectors](https://learn.arena.education/chapter4_alignment_science/4_persona_vectors/) | [notes](./daily_notes/day_091.md) | [code](./10_alignment/day_091_psych_probe.py) | [blog](./blog_posts/day_091.md) |

</details>

<details>
<summary><strong>Investigator Agents (Days 92–94)</strong></summary>

| Day | Topic | ARENA | Notes | Implementation | Blog |
|---|---|---|---|---|---|
| 92 | Investigator Agent Setup | [4.5 Investigator Agents](https://learn.arena.education/chapter4_alignment_science/5_investigator_agents/) | [notes](./daily_notes/day_092.md) | [code](./10_alignment/day_092_inv_agent.py) | [blog](./blog_posts/day_092.md) |
| 93 | Automated Investigation Experiments | [4.5 Investigator Agents](https://learn.arena.education/chapter4_alignment_science/5_investigator_agents/) | [notes](./daily_notes/day_093.md) | [code](./10_alignment/day_093_inv_experiments.py) | [blog](./blog_posts/day_093.md) |
| 94 | Investigator Agent Evaluation | [4.5 Investigator Agents](https://learn.arena.education/chapter4_alignment_science/5_investigator_agents/) | [notes](./daily_notes/day_094.md) | [code](./10_alignment/day_094_inv_eval.py) | [blog](./blog_posts/day_094.md) |

</details>

<details>
<summary><strong>Independent Research & Capstone (Days 95–100)</strong></summary>

| Day | Topic | Notes | Blog |
|---|---|---|---|
| 95 | Deep Dive: Deceptive Alignment | [notes](./daily_notes/day_095.md) | [blog](./blog_posts/day_095.md) |
| 96 | Deep Dive: Scalable Oversight | [notes](./daily_notes/day_096.md) | [blog](./blog_posts/day_096.md) |
| 97 | Independent Experiment — Design + Run | [notes](./daily_notes/day_097.md) | [blog](./blog_posts/day_097.md) |
| 98 | Independent Experiment — Analysis | [notes](./daily_notes/day_098.md) | [blog](./blog_posts/day_098.md) |
| 99 | Full Curriculum Review + Research Gaps | [notes](./daily_notes/day_099.md) | [blog](./blog_posts/day_099.md) |
| 100 | Capstone: Research Proposal | [notes](./daily_notes/day_100.md) | [blog](./blog_posts/day_100.md) |

</details>

---

## Repository Structure

```
arena-mechanistic-learning/
│── 01_backprops/ 
│   ├── 00_preequisites/          # Days 1–2   · PyTorch, einops, linear algebra
│   ├── 01_ray_tracing/            # Days 3–5   · Batched ops, 3D rendering
│   ├── 02_cnns_resnets/           # Days 6–8   · Convolutions, ResNet, CIFAR-10
│   ├── 03_optimization/           # Days 9–10  · SGD, Adam, W&B, sweeps
│   ├── 04_backprop/               # Days 11–12 · Autograd engine from scratch
│   ├── 05_vaes_gans/              # Days 13–14 · VAEs, GANs, generative models
│
├── 06_rl/                     # Days 15–29 · Bandits, Q-learning, DQN, PPO, RLHF
│
├── 07_transformers/           # Days 30–34 · GPT-2 from scratch, sampling
├── 08_mech_interp/            # Days 35–70 · TransformerLens, probes, SAEs,
│                              #              circuits, IOI, grokking, OthelloGPT
│
├── 09_evals/                  # Days 71–82 · Threat models, Inspect, LLM agents
│
├── 10_alignment/              # Days 83–100 · Emergent misalignment, reasoning
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


## Key Papers to read 

  


  
