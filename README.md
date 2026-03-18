# Property-Targeted-Inverse-Design-of-Stable-Inorganic-Crystals-via-LoRA
### Parameter-Efficient Fine-Tuning of Diffusion Models for Stable Crystal Structure Generation

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository provides the complete code, data, and reproduction instructions for the paper:

> **"GemNet-T Diffusion with Low-Rank Adaptation for Efficient Generative Design of Stable Crystals"**  
> Gourab Datta, Sarah Sharif, Yaser Banad — University of Oklahoma

We adapt a pre-trained crystal-structure diffusion model (base generator) with **Low-Rank Adaptation (LoRA)**, achieving 88.6 % of the FiLM baseline S·U·N score with **19× fewer trainable parameters** (3.2 M vs 62.2 M). DFT+U validation with VASP confirms 12 of 41 generated structures lie below the Materials Project convex hull, including novel intermetallics and rare-earth compounds.

---

## Table of Contents

1. [Repository Structure](#1-repository-structure)
2. [Requirements and Installation](#2-requirements-and-installation)
3. [Applying LoRA Modifications to the Base Generator](#3-applying-lora-modifications-to-the-base-generator)
4. [Step-by-Step Reproduction Guide](#4-step-by-step-reproduction-guide)
   - [Step 1 — Download Dataset](#step-1--download-the-dataset)
   - [Step 2 — Train LoRA Adapters](#step-2--train-lora-adapters)
   - [Step 3 — Generate Candidate Structures](#step-3--generate-candidate-structures)
   - [Step 4 — Evaluate S·U·N Metrics](#step-4--evaluate-sun-metrics)
   - [Step 5 — Analyse LoRA Weight Updates](#step-5--analyse-lora-weight-updates)
   - [Step 6 — Analyse Compositional Bias](#step-6--analyse-compositional-bias)
   - [Step 7 — Compute Efficiency Metrics](#step-7--compute-efficiency-metrics)
   - [Step 8 — DFT Validation (VASP)](#step-8--dft-validation-vasp)
   - [Step 9 — Regenerate Manuscript Figures](#step-9--regenerate-manuscript-figures)
   - [One-Command Reproduction](#one-command-reproduction)
5. [Dataset](#5-dataset)
6. [Model Checkpoints](#6-model-checkpoints)
7. [DFT Validation Results](#7-dft-validation-results)
8. [Key Results](#8-key-results)
9. [Citation](#9-citation)

---

## 1. Repository Structure

```
Property-Targeted-Inverse-Design-of-Stable-Inorganic-Crystals-via-LoRA/
│
├── README.md                      ← This file
├── requirements.txt               ← Python dependencies
├── environment.yml                ← Conda environment specification
│
├── src/lora/                      ← LoRA implementation (copy into base generator)
│   ├── __init__.py
│   ├── adapter.py                 ← Core LoRA primitives (LoRALayer, LoRAAdapter, etc.)
│   ├── gemnet_ctrl.py             ← GemNet-T + LoRA controller
│   └── generator_lora.py         ← LoRA-based denoiser adapter
│
├── training/
│   ├── train_rank8.py             ← Train LoRA Rank-8
│   ├── train_rank16.py            ← Train LoRA Rank-16  [paper main model]
│   └── train_rank32.py            ← Train LoRA Rank-32
│
├── generation/
│   └── generate_materials.py      ← Generate structures from any trained checkpoint
│
├── analysis/
│   ├── evaluate_sun.py            ← Compute S·U·N metrics and confidence intervals
│   ├── weight_analysis.py         ← SVD effective-rank analysis of LoRA weight updates
│   ├── composition_bias.py        ← Chemical composition bias test (LoRA vs FiLM)
│   └── efficiency_metrics.py      ← Parameter efficiency calculations
│
├── figures/
│   └── regenerate_figures.py      ← Reproduce all manuscript figures
│
├── data/
│   ├── download_dataset.py        ← Download Alexandria-MP-20 from HuggingFace
│   ├── results/                   ← DFT+U (VASP) validation results
│   │   ├── all_materials_summary.csv
│   │   ├── hull_analysis.csv
│   │   └── formation_energies.csv
│   ├── vasp_inputs/               ← POSCAR / CIF files for VASP calculations
│   ├── lora_seed_1_relaxed.extxyz ← MatterSim-relaxed LoRA structures (seed 1)
│   ├── lora_seed_2_relaxed.extxyz ← MatterSim-relaxed LoRA structures (seed 2)
│   ├── lora_seed_3_relaxed.extxyz ← MatterSim-relaxed LoRA structures (seed 3)
│   ├── film_seed_1_relaxed.extxyz ← MatterSim-relaxed FiLM structures (seed 1)
│   ├── film_seed_2_relaxed.extxyz ← MatterSim-relaxed FiLM structures (seed 2)
│   └── film_seed_3_relaxed.extxyz ← MatterSim-relaxed FiLM structures (seed 3)
│
└── scripts/
    └── reproduce_all.sh           ← End-to-end reproduction (runs all steps)
```

---

## 2. Requirements and Installation

### 2.1 Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate lora-crystal-generator
```

Or using pip in an existing Python 3.10+ environment:

```bash
pip install -r requirements.txt
```

### 2.2 Install the Base Generator

Our LoRA adapters are injected into the pre-trained **base crystal generator** (the GemNet-T diffusion model published by Microsoft as MatterGen). Clone and install it before running any scripts:

```bash
git clone https://github.com/microsoft/mattergen.git base_generator
cd base_generator
pip install -e .
cd ..
```

> **Note:** The base generator uses Hydra for configuration and PyTorch Lightning for training. Ensure compatible versions are installed as specified in `requirements.txt`.

### 2.3 Install MatterSim (for structure relaxation)

MatterSim is used to relax generated structures and predict energies. Install it from source:

```bash
pip install git+https://github.com/microsoft/mattersim.git
```

### 2.4 Verify Installation

```bash
python -c "import mattergen; print('Base generator OK')"
python -c "import mattersim; print('MatterSim OK')"
python -c "import pymatgen; print('pymatgen OK')"
```

---

## 3. Applying LoRA Modifications to the Base Generator

After installing the base generator, copy the three LoRA source files from `src/lora/` into the installed package. Replace `<base_generator_dir>` with the path to the cloned repository:

```bash
BASE_GENERATOR=./base_generator   # or wherever you cloned it

# 1. Core LoRA primitives
cp src/lora/adapter.py \
   $BASE_GENERATOR/mattergen/common/gemnet/lora_adapter.py

# 2. GemNet-T + LoRA controller
cp src/lora/gemnet_ctrl.py \
   $BASE_GENERATOR/mattergen/common/gemnet/gemnet_lora_ctrl.py

# 3. LoRA denoiser adapter (replaces FiLM adapter)
cp src/lora/generator_lora.py \
   $BASE_GENERATOR/mattergen/adapter_lora.py
```

**What these files do:**

| File | Class | Role |
|------|-------|------|
| `adapter.py` | `LoRALayer`, `LoRAAdapter`, `GemNetLoRAAdapter` | Core LoRA mathematics: $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$ |
| `gemnet_ctrl.py` | `GemNetTLoRACtrl` | Subclasses `GemNetT`; inserts `GemNetLoRAAdapter` into each GemNet block and overrides the forward pass |
| `generator_lora.py` | `GemNetTLoRAAdapter` | Subclasses `GemNetTDenoiser`; exposes `get_lora_parameters()`, `freeze_base_model()`, `merge_lora_weights()`, and `print_parameter_efficiency()` |

After copying, verify by importing:

```bash
python -c "from mattergen.adapter_lora import GemNetTLoRAAdapter; print('LoRA modifications applied OK')"
```

---

## 4. Step-by-Step Reproduction Guide

All commands are run from the **repository root** (`lora-crystal-generator/`).

---

### Step 1 — Download the Dataset

The Alexandria-MP-20 dataset (675,204 inorganic crystal structures, 86 elements) is available on Hugging Face:

```bash
python data/download_dataset.py --output_dir data/alex_mp_20
```

**Expected output:**
```
Downloading Alexandria-MP-20 from HuggingFace Hub → data/alex_mp_20
  Fetching split: train       → 540,162 structures saved
  Fetching split: validation  →  67,521 structures saved
  Fetching split: test        →  67,521 structures saved
Dataset download complete.
```

> **Disk space:** ~45 GB for the full dataset. Use `--subset 10000` for a quick test run.

**Dataset statistics:**

| Split      | Structures | Elements | Chemical classes |
|------------|-----------|----------|-----------------|
| Train      | 540,162   | 86       | Oxides, chalcogenides, pnictides, halides, intermetallics |
| Validation |  67,521   | —        | — |
| Test       |  67,521   | —        | — |

Structures are labelled with: formation energy, $E_{\text{hull}}$, band gap, and crystallographic metadata. All energies were computed with PBE/DFT.

---

### Step 2 — Train LoRA Adapters

Train each rank independently. Rank-16 is the main model reported in the paper.

```bash
# Rank 16 (paper main model — recommended to run first)
python training/train_rank16.py \
    --generator_path ./base_generator \
    --dataset_path   data/alex_mp_20 \
    --output_dir     checkpoints/lora_rank16

# Rank 8 (ablation)
python training/train_rank8.py \
    --generator_path ./base_generator \
    --dataset_path   data/alex_mp_20 \
    --output_dir     checkpoints/lora_rank8

# Rank 32 (ablation)
python training/train_rank32.py \
    --generator_path ./base_generator \
    --dataset_path   data/alex_mp_20 \
    --output_dir     checkpoints/lora_rank32
```

**Training hyperparameters (all ranks):**

| Hyperparameter | Rank 8 | Rank 16 | Rank 32 |
|----------------|--------|---------|---------|
| LoRA rank $r$ | 8 | 16 | 32 |
| LoRA $\alpha$ | 16 | 32 | 64 |
| Scaling $\alpha/r$ | 2.0 | 2.0 | 2.0 |
| Trainable params | 3,198,976 | 3,248,128 | 3,346,432 |
| Learning rate | 5×10⁻⁵ | 5×10⁻⁵ | 5×10⁻⁵ |
| Optimizer | Adam (β₁=0.9, β₂=0.999) | ← same | ← same |
| LR scheduler | ReduceLROnPlateau (factor 0.5, patience 20) | ← same | ← same |
| Batch size | 64/GPU | 64/GPU | 64/GPU |
| Weight decay | 0.01 | 0.01 | 0.01 |
| Grad clip | ±0.5 (value) | ±0.5 | ±0.5 |
| Max epochs | 200 | 200 | 200 |
| Early stopping | val/loss, patience 30 | ← same | ← same |
| **A** init | Kaiming uniform | ← same | ← same |
| **B** init | Zeros | ← same | ← same |
| Hardware | NVIDIA RTX 4090 (24 GB) | ← same | ← same |

> **Training time:** approximately 18–24 h for Rank-16 on a single RTX 4090 with the full Alexandria-MP-20 dataset.

**Early-stopping note:** All ranks were trained with a 200-epoch budget and early stopping.  
Actual stopping epochs vary per rank; check `checkpoints/lora_rankN/logs/` for learning curves.

---

### Step 3 — Generate Candidate Structures

```bash
# Rank 16
python generation/generate_materials.py \
    --checkpoint  checkpoints/lora_rank16/checkpoints/last.ckpt \
    --output_dir  generated/rank16 \
    --rank        16 \
    --num_samples 3000

# Rank 8
python generation/generate_materials.py \
    --checkpoint  checkpoints/lora_rank8/checkpoints/last.ckpt \
    --output_dir  generated/rank8 \
    --rank        8

# Rank 32
python generation/generate_materials.py \
    --checkpoint  checkpoints/lora_rank32/checkpoints/last.ckpt \
    --output_dir  generated/rank32 \
    --rank        32
```

**Generation settings (fixed for all runs):**

| Parameter | Value |
|-----------|-------|
| `energy_above_hull` target | 0.0 eV/atom |
| Guidance factor | 0.0 (disabled) |
| Diffusion steps | 1000 (DDPM) |
| Max atoms / cell | 20 |
| Seeds | 42, 123, 456 |
| Structures / run | 3,000 |

Post-processing (run automatically unless `--skip_postprocess` is passed):
1. Symmetry detection — spglib v2.0, tolerance 10⁻³ Å
2. Deduplication — `pymatgen.StructureMatcher` (0.25 Å, 5°)
3. MatterSim relaxation (MLFF energy minimization)
4. Thermodynamic stability via Materials Project API

**Expected valid structures after post-processing:**

| Model | Generated | Valid (post-processed) |
|-------|-----------|----------------------|
| LoRA Rank 8 | 3,000 | ~2,847 |
| LoRA Rank 16 | 3,000 | ~2,962 |
| LoRA Rank 32 | 3,000 | ~2,873 |
| FiLM (baseline) | 3,000 | ~2,188 |

---

### Step 4 — Evaluate S·U·N Metrics

```bash
python analysis/evaluate_sun.py \
    --results_dir generated/rank8 generated/rank16 generated/rank32 \
    --labels      "LoRA R8" "LoRA R16" "LoRA R32" \
    --output      analysis_results/sun_metrics.json \
    --plot
```

**Expected output:**

```
============================================================
Model            N       S       U       N      S·U·N   SE(S)
------------------------------------------------------------
LoRA R8       2847   0.472   0.966   0.735   0.3350  ±0.003
LoRA R16      2962   0.493   0.961   0.721   0.3410  ±0.014
LoRA R32      2873   0.475   0.968   0.724   0.3330  ±0.029
============================================================
```

Bootstrapped standard errors are computed with 1,000 resamples.

---

### Step 5 — Analyse LoRA Weight Updates

Performs SVD on the learned weight updates $\Delta W = BA$ to compute the **effective rank** (number of singular values needed to capture 90 % of the update energy):

```bash
python analysis/weight_analysis.py \
    --checkpoint_dir checkpoints/ \
    --output_dir     analysis_results/
```

**Expected finding:** All nominal ranks (8, 16, 32) converge to an effective rank of ~9, consistent with an information-bottleneck ceiling independent of nominal rank.

---

### Step 6 — Analyse Compositional Bias

```bash
python analysis/composition_bias.py \
    --generated_dir generated/ \
    --output_dir    analysis_results/
```

Tests whether the oxide fraction is statistically indistinguishable across LoRA ranks (expected: $\chi^2 = 0.14$, $p = 0.93$) and significantly different from the FiLM baseline (expected: 40.3 % vs 8.3 %, $\chi^2 > 1000$, $p < 10^{-400}$).

---

### Step 7 — Compute Efficiency Metrics

```bash
python analysis/efficiency_metrics.py \
    --checkpoint_dir checkpoints/ \
    --output_dir     analysis_results/
```

Computes trainable parameter counts, checkpoint sizes, and S·U·N per million trainable parameters for each rank and for the FiLM baseline.

---

### Step 8 — DFT Validation (VASP)

DFT+U calculations require a licensed VASP installation and are **not run automatically**. The converged results for all 41 structures are included in `data/results/`.

**To prepare new VASP inputs for your own candidate structures:**

```bash
# Select and prepare POSCAR/INCAR/KPOINTS for top candidates
# (Edit the script to point to your generated structures)
python analysis/weight_analysis.py --prepare_vasp_inputs \
    --candidates_dir generated/rank16 \
    --output_dir     data/vasp_inputs/
```

**VASP settings used in the paper:**

| Setting | Value |
|---------|-------|
| Version | VASP 6.3 |
| Pseudopotentials | PAW-PBE |
| Energy cutoff | 520 eV (1.3 × ENMAX) |
| k-points | ≥1000 per reciprocal atom, Γ-centred |
| Electronic convergence | 10⁻⁶ eV |
| Ionic convergence | < 0.01 eV/Å |
| Hubbard-U (Gd) | U = 6.7 eV, J = 0.7 eV |
| Hubbard-U (Eu) | U = 7.0 eV |
| Hull reference | pymatgen + raw MP energies (no MP2020Compat.) |

> **Important:** Hull energies in `data/results/` are computed without the MP2020Compatibility corrections applied by the Materials Project. This introduces a systematic offset of 0.05–0.10 eV/atom for oxide-rich compositions; see paper Section 2.5 for a full discussion.

---

### Step 9 — Regenerate Manuscript Figures

```bash
python figures/regenerate_figures.py
```

Figures are saved to `figure/` as both `.pdf` (for LaTeX) and `.png` (for preview):

| Figure | Description |
|--------|-------------|
| `Figure1_Parameter_Efficiency.pdf` | Pareto frontier (params vs S·U·N) + metric decomposition |
| `Figure2_Chemical_Diversity.pdf` | Compositional complexity, element frequencies, chemical classes |
| `Figure3_DFT_Validation.pdf` | Parity plot (MatterSim vs VASP) + energy distribution violins |
| `Figure4_Parameter_Efficiency_Metrics.pdf` | Trainable params, checkpoint sizes, S·U·N per param |
| `Architecture.pdf` | GemNet-T + LoRA pipeline schematic |

> **Note:** Figure 3 reads from `data/results/all_materials_summary.csv`. All other figures use data hard-coded from Tables 1–3 of the paper.

---

### One-Command Reproduction

To run all steps in sequence, edit the `CONFIGURATION` block at the top of `scripts/reproduce_all.sh` to set `GENERATOR_PATH`, then:

```bash
chmod +x scripts/reproduce_all.sh
bash scripts/reproduce_all.sh
```

The script honours `SKIP_*` flags to resume from any step after a failed run.

---

## 5. Dataset

**Alexandria-MP-20** — 675,204 inorganic crystal structures combining the Materials Project (MP-20) and the Alexandria database.

| Property | Details |
|----------|---------|
| Total structures | 675,204 |
| Training split | 540,162 |
| Validation split | 67,521 |
| Test split | 67,521 |
| Elements covered | 86 |
| Energy functional | PBE (DFT relaxation) |
| Labels | Formation energy, $E_{\text{hull}}$, band gap, crystallographic metadata |
| Chemical classes | Oxides, chalcogenides, pnictides, halides, intermetallics |
| Source | [HuggingFace: OMatG/Alex-MP-20](https://huggingface.co/datasets/OMatG/Alex-MP-20) |
| Reference | Miret et al. (2023) |

---

## 6. Model Checkpoints

Trained checkpoint files are not hosted in this repository due to size. To obtain the Rank-16 checkpoint used in the paper, either:

1. **Train from scratch** following Step 2 above (approximately 18–24 h on RTX 4090), or
2. **Contact the authors** for the checkpoint file.

The model configuration is in `configs/train_rank16.yaml`.

**Relaxed structure data** (MatterSim-processed, ready for S·U·N evaluation) are provided in `data/`:

```
data/lora_seed_1_relaxed.extxyz  — LoRA Rank-16 structures, seed 42
data/lora_seed_2_relaxed.extxyz  — LoRA Rank-16 structures, seed 123
data/lora_seed_3_relaxed.extxyz  — LoRA Rank-16 structures, seed 456
data/film_seed_1_relaxed.extxyz  — FiLM baseline structures, seed 42
data/film_seed_2_relaxed.extxyz  — FiLM baseline structures, seed 123
data/film_seed_3_relaxed.extxyz  — FiLM baseline structures, seed 456
```

These files can be used to reproduce the S·U·N evaluation without re-running generation.

---

## 7. DFT Validation Results

`data/results/all_materials_summary.csv` contains converged VASP DFT+U results for all 41 evaluated structures:

| Column | Description |
|--------|-------------|
| `formula` | Chemical formula |
| `n_atoms` | Atoms in the DFT unit cell |
| `delta_Hf_eV_per_atom` | Formation enthalpy (eV/atom) vs elemental phases |
| `e_above_hull_eV_per_atom` | Energy above Materials Project convex hull (eV/atom) |
| `stability_class` | `below_hull` / `stable` / `metastable` / `unstable` |
| `discovery_type` | T1 = new polymorph; T2 = new stoichiometry (robust); T3 = new stoichiometry (tentative) |

**Summary of DFT validation results:**

| Category | Count | Notable examples |
|----------|-------|-----------------|
| Below MP hull | 12 | Gd₃InC (−0.111), EuO (−0.078), GdRu (−0.053), LiGd₂Ir (−0.035) eV/atom |
| Stable (≤0.10 eV/atom) | 9 | Mo₂TaTi, Ta₃Ru, TiRe, Gd₂AgRu |
| Metastable (0.10–0.20) | 6 | W₂C, Ta₂C, EuHfO₃, HfN |
| Unstable (>0.20) | 14 | — |
| **Total converged** | **41** | 50 submitted; 8 pure-element excluded; 1 failed |

---

## 8. Key Results

| Model | Trainable params | Checkpoint | S | U | N | **S·U·N** | Rel. to FiLM |
|-------|-----------------|-----------|---|---|---|-----------|--------------|
| **LoRA Rank 16** (ours) | **3.25 M** | **189 MB** | 0.493 | 0.961 | 0.721 | **0.341** | **88.6 %** |
| LoRA Rank 8 | 3.20 M | 189 MB | 0.472 | 0.966 | 0.735 | 0.335 | 87.0 % |
| LoRA Rank 32 | 3.35 M | 190 MB | 0.475 | 0.968 | 0.724 | 0.333 | 86.5 % |
| FiLM baseline | 62.19 M | 537 MB | 0.536 | 1.000 | 0.718 | 0.385 | 100 % |

**Compositional bias (oxide fraction):**

| Model group | Oxide fraction | χ² vs FiLM |
|-------------|---------------|-----------|
| LoRA (all ranks, pooled) | 40.3 % ± 0.8 % | χ² = 1,847, p < 10⁻⁴⁰⁰ |
| LoRA rank independence | — | χ² = 0.14, p = 0.93 |
| FiLM baseline | 8.3 % | — |

---

## 9. Citation

If you use this code or data in your research, please cite:

```bibtex
@article{datta2025lora_crystal,
  title   = {GemNet-T Diffusion with Low-Rank Adaptation for Efficient
             Generative Design of Stable Crystals},
  author  = {Datta, Gourab and Sharif, Sarah and Banad, Yaser},
  journal = {npj Computational Materials},
  year    = {2025},
  note    = {Under review}
}
```

---

## Licence

This repository is released under the MIT Licence. See [LICENSE](LICENSE) for details.

The base crystal generator is copyright Microsoft Corporation and is licensed separately under the MIT Licence at its own repository.
