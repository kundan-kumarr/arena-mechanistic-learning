
# ARENA Mechanistic Learning  
**100 Days of Mechanistic Learning (ARENA + My Notes, Implementations, and Experiments)**  
  
Based on: [ARENA – Alignment Research Engineer Accelerator](https://learn.arena.education/)  
  
## Overview  
  
This repository is my public learning log for the **ARENA (Alignment Research Engineer Accelerator)** curriculum.  
  
It contains my:  
  
- Personal notes and explanations (my understanding in my own words)  
- Mathematical derivations (where helpful)  
- Implementations from scratch (NumPy / minimal PyTorch)  
- Small experiments and sanity checks (e.g., gradient checks, ablations)  
- Blog posts summarizing what I learned and how I think about it  
  
The goal is to build **mechanistic understanding**—not just run notebooks—starting from basic neural network fundamentals and progressing toward transformers, reinforcement learning, and alignment-related topics.  

## 100 Days of Mechanistic Learning Challenge

This repository is part of my **100 Days of Mechanistic Learning Challenge**.

The challenge is simple:

**Every day, I study one concept from ARENA and publish my understanding as code + notes (and often a blog post).**

This is not a claim of novel research results.  
It is a public record of learning, implementation, and reflection.

Daily deliverables typically include:

- A short written explanation in my own words
- A working implementation (minimal and readable)
- At least one check or experiment (sanity check, visualization, or ablation)
- Optional: a blog post that compresses the idea into a clear narrative

---

## What I’m Learning (high-level progression)

This repo follows the ARENA curriculum, roughly in this progression:

**Fundamentals → Transformers → Reinforcement Learning → Interpretability / Alignment**

Within each topic, my workflow is:

**Derive → Implement → Verify → Experiment → Explain (Blog/Notes)**

---

## Motivation

Modern ML frameworks automate gradient computation. While powerful, that abstraction can hide the underlying mechanism of learning.

At its core, learning in neural networks consists of:

> Routing responsibility signals (gradients) backward through a computational graph.

By deriving and implementing core algorithms directly, I aim to understand:

- how gradients flow (and fail)
- how representations form
- how policies learn over time in RL
- how these mechanisms relate to alignment and safety

---

## Attribution and Credit

This repository is based on my independent implementations and notes while studying the ARENA curriculum.

Full credit goes to the ARENA team for creating and providing this educational program and materials.

Official ARENA resources:

- Curriculum: https://learn.arena.education/  
- Website: https://www.arena.education/  

ARENA’s curriculum includes:

- Neural network fundamentals
- Backpropagation and optimization
- Transformers
- Reinforcement learning and RLHF
- Mechanistic interpretability
- Alignment research methods

All original course materials belong to ARENA.

This repository contains only my personal:

- implementations
- derivations
- explanations
- experiments/sanity checks
- blog posts and reflections

This repository is not an official ARENA resource.

---
## Repository Structure

```text
arena-mechanistic-learning/

01_backprop/
  Backprop derivations + NumPy implementations + gradient checks

02_transformers/
  Attention + transformer components + experiments

03_reinforcement_learning/
  Policy gradients, PPO, actor-critic + diagnostics

04_interpretability/
  Probing, activation analysis, mechanistic studies

05_alignment_notes/
  Alignment-related notes and reflections (non-claims, learning-focused)

blog_posts/
  Quarto/Markdown posts derived from my notes and implementations

daily_logs/
  Short daily entries for the 100-day challenge



