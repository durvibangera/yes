import argparse
import sys
import subprocess
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MainDiagnostic")

def run_script(script_path: str, args: list):
    cmd = [sys.executable, script_path] + args
    logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.error(f"Script {script_path} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Retrieval Diagnostic Tool Runner")
    parser.add_argument("--project_id", type=str, required=True, help="Project ID or comma-separated list of Project IDs to analyze")
    parser.add_argument("--top_k", type=int, default=10, help="Retrieval Top-K")
    args = parser.parse_args()

    project_ids = [p.strip() for p in args.project_id.split(",")]
    top_k = args.top_k

    # Paths to diagnostic steps
    diagnostics_dir = os.path.join(os.path.dirname(__file__), "diagnostics")
    step1 = os.path.join(diagnostics_dir, "run_all_strategies.py")
    step2 = os.path.join(diagnostics_dir, "overlap_calculator.py")
    step3 = os.path.join(diagnostics_dir, "bucket_analysis.py")
    step4 = os.path.join(diagnostics_dir, "report_generator.py")

    from ingestion.config import settings
    import shutil

    for project_id in project_ids:
        logger.info("=" * 60)
        logger.info(f"STARTING RETRIEVAL DIAGNOSTIC RUN FOR PROJECT: {project_id}")
        logger.info("=" * 60)

        # Flag ABS Standards usage in training warning
        if project_id == "ABS Standards":
            logger.warning("=" * 60)
            logger.warning("NOTICE: Running diagnostic on 'ABS Standards'.")
            logger.warning("Verify that ABS Standards data has not been used during model training/tuning.")
            logger.warning("=" * 60)

        # Step 1: Run all strategies
        run_script(step1, ["--project_id", project_id, "--top_k", str(top_k)])

        # Step 2: Overlap calculation (Jaccard + Cosine)
        run_script(step2, ["--project_id", project_id])

        # Step 3: Bucket analysis
        run_script(step3, [])

        # Step 4: Report generation
        run_script(step4, [])

        # Copy outputs to project-specific names to prevent overwriting
        project_clean = project_id.replace(" ", "_")
        shutil.copy(os.path.join(settings.DATA_ROOT, "outputs", "overlap_scores.jsonl"), os.path.join(settings.DATA_ROOT, "outputs", f"{project_clean}_overlap_scores.jsonl"))
        shutil.copy(os.path.join(settings.DATA_ROOT, "outputs", "bucket_summary.json"), os.path.join(settings.DATA_ROOT, "outputs", f"{project_clean}_bucket_summary.json"))
        shutil.copy(os.path.join(settings.DATA_ROOT, "outputs", "inspection_report.md"), os.path.join(settings.DATA_ROOT, "outputs", f"{project_clean}_inspection_report.md"))

        logger.info(f"Project {project_id} diagnostic run completed. Saved copy to {project_clean}_ prefix.")

    logger.info("=" * 60)
    logger.info("ALL DIAGNOSTIC RUNS COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
