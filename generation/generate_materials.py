#!/usr/bin/env python3
"""
Generate candidate crystal structures from a trained LoRA adapter.

Conditions all generation runs on energy_above_hull = 0.0 eV/atom with
classifier-free guidance disabled, matching the experimental setup in the paper.

Usage
-----
  # Generate 3 000 structures from the Rank-16 checkpoint
  python generation/generate_materials.py \
      --checkpoint   checkpoints/lora_rank16/checkpoints/last.ckpt \
      --output_dir   generated/rank16 \
      --rank         16 \
      --num_samples  3000

  # Run all three ranks in sequence
  for RANK in 8 16 32; do
      python generation/generate_materials.py \\
          --checkpoint checkpoints/lora_rank${RANK}/checkpoints/last.ckpt \\
          --output_dir generated/rank${RANK} \\
          --rank       ${RANK}
  done
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Generation defaults (matching paper settings)
DEFAULT_NUM_SAMPLES = 3000
DEFAULT_NUM_STEPS = 1000          # DDPM schedule
DEFAULT_MAX_ATOMS = 20
EHULL_TARGET = 0.0                # eV/atom — target thermodynamically stable
GUIDANCE_FACTOR = 0.0             # classifier-free guidance disabled


def run_generation(
    checkpoint: Path,
    output_dir: Path,
    rank: int,
    num_samples: int,
    seeds: list[int],
    generator_path: Path,
) -> None:
    """Run the generation pipeline for a single checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        seed_out = output_dir / f"seed_{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "-m", "mattergen.run_eval",
            f"checkpoint_path={checkpoint}",
            f"output_path={seed_out}",
            f"num_samples={num_samples // len(seeds)}",
            f"properties_to_condition_on={{energy_above_hull:{EHULL_TARGET}}}",
            f"diffusion_guidance_factor={GUIDANCE_FACTOR}",
            f"num_steps={DEFAULT_NUM_STEPS}",
            f"max_atoms={DEFAULT_MAX_ATOMS}",
            f"seed={seed}",
        ]

        logger.info(f"Generating with seed {seed} → {seed_out}")
        logger.info("Command: " + " ".join(cmd))

        env = os.environ.copy()
        if generator_path:
            env["PYTHONPATH"] = str(generator_path) + os.pathsep + env.get("PYTHONPATH", "")

        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            logger.error(f"Generation failed for seed {seed}")
            sys.exit(result.returncode)

    logger.info(f"Generation complete. Results in {output_dir}")


def post_process(output_dir: Path, generator_path: Path) -> None:
    """
    Run standard post-processing on generated structures:
      - Symmetry detection (spglib)
      - Deduplication (pymatgen StructureMatcher)
      - MatterSim relaxation
      - Thermodynamic stability via Materials Project API
    """
    cmd = [
        sys.executable, "-m", "mattergen.evaluation.evaluate",
        f"results_path={output_dir}",
        "run_relaxation=true",
        "run_uniqueness=true",
        "run_novelty=true",
        "relaxation_backend=mattersim",
    ]

    logger.info("Running post-processing ...")
    env = os.environ.copy()
    if generator_path:
        env["PYTHONPATH"] = str(generator_path) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        logger.warning(
            "Post-processing step failed (non-zero exit). "
            "Raw structures are still in the output directory."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate crystal structures from a trained LoRA adapter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to the trained LoRA checkpoint (.ckpt file).",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("generated/output"),
        help="Directory to save generated structures.",
    )
    parser.add_argument(
        "--rank", type=int, choices=[8, 16, 32], default=16,
        help="LoRA rank of the checkpoint being used (for logging only).",
    )
    parser.add_argument(
        "--num_samples", type=int, default=DEFAULT_NUM_SAMPLES,
        help="Total number of structures to generate.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456],
        help="Random seeds (structures split equally across seeds).",
    )
    parser.add_argument(
        "--generator_path", type=Path, default=None,
        help="Path to generator package root (added to PYTHONPATH if provided).",
    )
    parser.add_argument(
        "--skip_postprocess", action="store_true",
        help="Skip the post-processing step (relaxation, deduplication, etc.).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger.info(f"Generating {args.num_samples} structures from LoRA Rank-{args.rank}")
    logger.info(f"Checkpoint : {args.checkpoint}")
    logger.info(f"Output dir : {args.output_dir}")
    logger.info(f"EaH target : {EHULL_TARGET} eV/atom")
    logger.info(f"Seeds      : {args.seeds}")

    run_generation(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        rank=args.rank,
        num_samples=args.num_samples,
        seeds=args.seeds,
        generator_path=args.generator_path,
    )

    if not args.skip_postprocess:
        post_process(args.output_dir, args.generator_path)
