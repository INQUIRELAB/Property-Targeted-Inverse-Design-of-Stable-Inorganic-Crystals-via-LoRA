#!/usr/bin/env bash
# =============================================================================
# reproduce_all.sh  —  End-to-end reproduction script
#
# Runs every step of the pipeline in sequence:
#   1. Download dataset
#   2. Train LoRA adapters (ranks 8, 16, 32)
#   3. Generate candidate structures
#   4. Evaluate S·U·N metrics
#   5. Analyse LoRA weight updates (SVD)
#   6. Analyse compositional bias
#   7. Compute efficiency metrics
#   8. Regenerate manuscript figures
#
# Prerequisites
# -------------
#   conda activate lora-crystal-generator   (or equivalent venv)
#   pip install -r requirements.txt
#   pip install git+https://github.com/microsoft/mattergen.git
#   pip install git+https://github.com/microsoft/mattersim.git
#   Apply LoRA modifications (see README §3)
#
# Configuration
# -------------
# Edit the variables in the CONFIGURATION block below before running.
#
# Usage
# -----
#   chmod +x scripts/reproduce_all.sh
#   bash scripts/reproduce_all.sh
# =============================================================================

set -euo pipefail
IFS=$'\n\t'

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
GENERATOR_PATH=""          # Path to installed base generator package root
DATASET_PATH="data/alex_mp_20"
CHECKPOINT_DIR="checkpoints"
GENERATED_DIR="generated"
RESULTS_DIR="analysis_results"
FIGURES_DIR="figure"
NUM_SAMPLES=3000
SEEDS="42 123 456"

# Set to 1 to skip a step (useful when resuming a failed run)
SKIP_DOWNLOAD=0
SKIP_TRAINING=0
SKIP_GENERATION=0
SKIP_EVALUATION=0
SKIP_ANALYSIS=0
SKIP_FIGURES=0
# ─────────────────────────────────────────────────────────────────────────────

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

[[ -z "$GENERATOR_PATH" ]] && die "Set GENERATOR_PATH in scripts/reproduce_all.sh before running."
[[ ! -d "$GENERATOR_PATH" ]] && die "GENERATOR_PATH '$GENERATOR_PATH' does not exist."

mkdir -p "$RESULTS_DIR" "$GENERATED_DIR" "$CHECKPOINT_DIR" "$FIGURES_DIR"

# ─── STEP 1: Download dataset ─────────────────────────────────────────────────
if [[ $SKIP_DOWNLOAD -eq 0 ]]; then
    log "STEP 1 — Downloading Alexandria-MP-20 dataset"
    python data/download_dataset.py --output_dir "$DATASET_PATH"
else
    log "STEP 1 — Skipped (SKIP_DOWNLOAD=1)"
fi

# ─── STEP 2: Train LoRA adapters ─────────────────────────────────────────────
if [[ $SKIP_TRAINING -eq 0 ]]; then
    for RANK in 8 16 32; do
        log "STEP 2.$RANK — Training LoRA Rank-$RANK adapter"
        python "training/train_rank${RANK}.py" \
            --generator_path "$GENERATOR_PATH" \
            --dataset_path   "$DATASET_PATH" \
            --output_dir     "$CHECKPOINT_DIR/lora_rank${RANK}"
    done
else
    log "STEP 2 — Skipped (SKIP_TRAINING=1)"
    log "         Make sure checkpoints exist under $CHECKPOINT_DIR/"
fi

# ─── STEP 3: Generate candidate structures ───────────────────────────────────
if [[ $SKIP_GENERATION -eq 0 ]]; then
    for RANK in 8 16 32; do
        log "STEP 3.$RANK — Generating $NUM_SAMPLES structures with Rank-$RANK"
        CKPT="$CHECKPOINT_DIR/lora_rank${RANK}/checkpoints/last.ckpt"
        [[ ! -f "$CKPT" ]] && die "Checkpoint not found: $CKPT"
        python generation/generate_materials.py \
            --checkpoint     "$CKPT" \
            --output_dir     "$GENERATED_DIR/rank${RANK}" \
            --rank           "$RANK" \
            --num_samples    "$NUM_SAMPLES" \
            --seeds          $SEEDS \
            --generator_path "$GENERATOR_PATH"
    done
else
    log "STEP 3 — Skipped (SKIP_GENERATION=1)"
fi

# ─── STEP 4: Evaluate S·U·N metrics ─────────────────────────────────────────
if [[ $SKIP_EVALUATION -eq 0 ]]; then
    log "STEP 4 — Computing S·U·N metrics"
    python analysis/evaluate_sun.py \
        --results_dir "$GENERATED_DIR/rank8" \
                      "$GENERATED_DIR/rank16" \
                      "$GENERATED_DIR/rank32" \
        --labels      "LoRA R8" "LoRA R16" "LoRA R32" \
        --output      "$RESULTS_DIR/sun_metrics.json" \
        --plot
else
    log "STEP 4 — Skipped (SKIP_EVALUATION=1)"
fi

# ─── STEP 5: Analyse LoRA weight updates ─────────────────────────────────────
if [[ $SKIP_ANALYSIS -eq 0 ]]; then
    log "STEP 5a — Analysing LoRA weight updates (SVD effective rank)"
    python analysis/weight_analysis.py \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --output_dir     "$RESULTS_DIR"

    log "STEP 5b — Analysing compositional bias"
    python analysis/composition_bias.py \
        --generated_dir "$GENERATED_DIR" \
        --output_dir    "$RESULTS_DIR"

    log "STEP 5c — Computing parameter efficiency metrics"
    python analysis/efficiency_metrics.py \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --output_dir     "$RESULTS_DIR"
else
    log "STEP 5 — Skipped (SKIP_ANALYSIS=1)"
fi

# ─── STEP 6: Regenerate figures ───────────────────────────────────────────────
if [[ $SKIP_FIGURES -eq 0 ]]; then
    log "STEP 6 — Regenerating manuscript figures"
    python figures/regenerate_figures.py
else
    log "STEP 6 — Skipped (SKIP_FIGURES=1)"
fi

# ─── Done ─────────────────────────────────────────────────────────────────────
log "============================================================"
log "Reproduction complete."
log "  S·U·N metrics  : $RESULTS_DIR/sun_metrics.json"
log "  Analysis plots : $RESULTS_DIR/"
log "  Figures        : $FIGURES_DIR/"
log "============================================================"
