# run_all.py

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


ALL_SCRIPTS_DIR = Path(__file__).resolve().parent
SANDBOX_DIRNAME = "example_2"
OUTPUTS_DIRNAME = "outputs"


def _ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _run_script(script_name: str) -> None:
    script_path = ALL_SCRIPTS_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    result = subprocess.run(
        [sys.executable, str(script_path.name)],
        cwd=str(ALL_SCRIPTS_DIR),
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr)
        raise RuntimeError(f"Script failed: {script_name} (exit={result.returncode})")


def _convert_merged_report_to_slide_results(merged_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert pipeline output into the same shape as compliance_engine.run_compliance_analysis.

    The UI expects a list of slides like:
      { slide_id, slide_type, conforme, violations: [{rule_id, rule_text, issue, evidence, suggested_fix}], notes }
    """
    results: List[Dict[str, Any]] = []

    slides = merged_report.get("slides_analysis", []) if isinstance(merged_report, dict) else []
    for slide in slides:
        slide_num = slide.get("slide_number")
        if slide_num is None:
            continue

        violations = slide.get("violations", []) or []
        formatted_violations: List[Dict[str, Any]] = []
        for v in violations:
            formatted_violations.append(
                {
                    "rule_id": v.get("rule_id", ""),
                    "rule_text": v.get("rule_text", ""),
                    "issue": v.get("explanation", v.get("violation_type", "")),
                    "evidence": v.get("violating_text", v.get("violating_content", "")),
                    "suggested_fix": v.get("correction", ""),
                }
            )

        results.append(
            {
                "slide_id": int(slide_num),
                "slide_type": slide.get("slide_title", ""),
                "conforme": len(formatted_violations) == 0,
                "violations": formatted_violations,
                "notes": "",
            }
        )

    results.sort(key=lambda r: r.get("slide_id", 0))
    return results


def run_pipeline(pptx_path: str, prospectus_path: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Main entrypoint for Flask.

    Creates/overwrites the sandbox folders:
      all_scripts/example_2
      all_scripts/example_2/outputs
    Copies the provided files and writes metadata.json, then runs the pipeline.
    Returns slide-level results compatible with templates/results.html and pptx_annotator.
    """
    pptx_src = Path(pptx_path).resolve()
    prospectus_src = Path(prospectus_path).resolve()

    if not pptx_src.exists():
        raise FileNotFoundError(f"PPTX not found: {pptx_src}")
    if not prospectus_src.exists():
        raise FileNotFoundError(f"Prospectus not found: {prospectus_src}")

    sandbox_dir = ALL_SCRIPTS_DIR / SANDBOX_DIRNAME
    outputs_dir = sandbox_dir / OUTPUTS_DIRNAME

    _ensure_empty_dir(sandbox_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Copy inputs to expected filenames
    shutil.copy2(pptx_src, sandbox_dir / "expl2.pptx")
    shutil.copy2(prospectus_src, sandbox_dir / f"prospectus{prospectus_src.suffix.lower()}")

    metadata_path = sandbox_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Run the scripts in order (they are hardcoded to use example_2/...)
    for script in [
        "pptx_chunker1.py",
        "metadata_enriched_1.py",
        "metadata_enriched_2.py",
        "metadata_enriched_3.py",
        "filtrage_des_regles.py",
        "analyse_slide.py",
        "2_2.py",
        "2_25.py",
        "2_3.py",
        "2_31.py",
        "fusion.py",
    ]:
        _run_script(script)

    merged_report_path = outputs_dir / "merged_compliance_report.json"
    if not merged_report_path.exists():
        raise FileNotFoundError(f"Pipeline finished but merged report missing: {merged_report_path}")

    with open(merged_report_path, "r", encoding="utf-8") as f:
        merged_report = json.load(f)

    return _convert_merged_report_to_slide_results(merged_report)


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 3:
        print("Usage: python run_all.py <pptx_path> <prospectus_path> '<metadata_json_string>'")
        return 2

    pptx_path = argv[0]
    prospectus_path = argv[1]
    try:
        metadata = json.loads(argv[2])
    except Exception as e:
        raise ValueError("metadata_json_string must be valid JSON") from e

    run_pipeline(pptx_path, prospectus_path, metadata)
    print("ALL DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
