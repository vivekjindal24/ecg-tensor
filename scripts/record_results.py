import json
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import mlflow
from mlflow.tracking import MlflowClient


ONE_SHOT = True  # set True to run once and exit; set False only when you want loop behavior


EXPERIMENT_NAME = "ECG_Tensor_Research"
if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
    print(f"Experiment '{EXPERIMENT_NAME}' not found. Creating it...")
    mlflow.create_experiment(
        name=EXPERIMENT_NAME,
        artifact_location="./artifacts/mlflow"
    )

ROOT = Path(r"D:\ecg-research")
LOGS = ROOT / "logs"
LATEX_FILE = ROOT / "paper" / "ECG_Tensor_Research_Paper.tex"
HISTORY = LOGS / "experiment_history.json"

TRACKING_URI = "http://127.0.0.1:8000"
EXPERIMENT_NAME = "ECG_Tensor_Research"

def fetch_latest_run():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not exp:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found.")
    runs = client.search_runs(exp.experiment_id, order_by=["attribute.start_time DESC"], max_results=1)
    if not runs:
        return None
    run = runs[0]
    return {
        "run_id": run.info.run_id,
        "params": run.data.params,
        "metrics": run.data.metrics,
        "artifact_uri": run.info.artifact_uri
    }

def append_to_history(entry):
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    if not HISTORY.exists():
        with HISTORY.open("w", encoding="utf-8") as f:
            json.dump([], f)
    with HISTORY.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.append(entry)
    with HISTORY.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def _ensure_latex_document() -> None:
    LATEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    if LATEX_FILE.exists():
        return
    base = (
        "% Auto-generated ECG Tensor Research manuscript\n"
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{float}\n"
        "\\begin{document}\n"
        "\\section*{ECG Tensor Research Overview}\n"
        "This document aggregates experimental findings.\\\n"
        "\\textit{Preliminary results; verify before submission.}\\\n"
        "\\bigskip\n"
        "\\end{document}\n"
    )
    LATEX_FILE.write_text(base, encoding="utf-8")


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\\textbackslash{}",
        "&": r"\\&",
        "%": r"\\%",
        "$": r"\\$",
        "#": r"\\#",
        "_": r"\\_",
        "{": r"\\{",
        "}": r"\\}",
        "~": r"\\textasciitilde{}",
        "^": r"\\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _artifact_directory(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        path = parsed.path.lstrip("/")
        return Path(path)
    return Path(uri)


def _find_roc_figure(dir_path: Path) -> Path | None:
    if not dir_path.exists():
        return None
    for pattern in ("*roc*", "*ROC*"):
        for ext in (".png", ".pdf", ".svg"):
            matches = list(dir_path.rglob(pattern + ext))
            if matches:
                return matches[0]
    return None


def update_latex(entry):
    _ensure_latex_document()

    params = entry.get("params", {})
    metrics = entry.get("metrics", {})
    artifact_uri = entry.get("artifact_uri", "")
    artifact_dir = _artifact_directory(artifact_uri) if artifact_uri else None
    roc_fig = _find_roc_figure(artifact_dir) if artifact_dir else None
    roc_tex_path = _latex_escape(roc_fig.as_posix()) if roc_fig else "path/to/roc_curve.png"

    params_str = ", ".join(f"{k}={v}" for k, v in sorted(params.items())) or "None"

    section = [
        f"\\section*{{New Experiment Results - {_latex_escape(entry['timestamp'])}}}",
            f"\\textbf{{Run ID:}} {_latex_escape(entry['run_id'])} \\\\",
            f"\\textbf{{Parameters:}} {_latex_escape(params_str)} \\\\",
        "\\textbf{Metrics:}",
        "\\begin{itemize}",
        f"  \\item Accuracy: {_latex_escape(str(metrics.get('accuracy', 'N/A')))}",
        f"  \\item F1 Score: {_latex_escape(str(metrics.get('f1', 'N/A')))}",
        f"  \\item AUROC: {_latex_escape(str(metrics.get('auroc', 'N/A')))}",
        f"  \\item Sensitivity: {_latex_escape(str(metrics.get('sensitivity', 'N/A')))}",
        f"  \\item Specificity: {_latex_escape(str(metrics.get('specificity', 'N/A')))}",
        "\\end{itemize}",
        "",
        "\\textbf{Figures:}",
        "\\begin{figure}[H]",
        "  \\centering",
        f"  \\includegraphics[width=0.8\\textwidth]{{{roc_tex_path}}}",
        "  \\caption{ROC Curve for current run.}",
        "\\end{figure}",
        "",
        "\\textit{Preliminary results; verify before submission.}",
        "\\bigskip",
        ""
    ]
    section_text = "\n".join(section)

    existing = LATEX_FILE.read_text(encoding="utf-8")
    if "\\end{document}" in existing:
        updated = existing.replace("\\end{document}", section_text + "\n\\end{document}")
    else:
        updated = existing + "\n" + section_text
    LATEX_FILE.write_text(updated, encoding="utf-8")
    print(f"✅ LaTeX manuscript updated at: {LATEX_FILE}")


def process_run(run: dict) -> None:
    run = dict(run)
    run["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("Appending to experiment_history.json...")
    append_to_history(run)

    print("Updating research paper...")
    update_latex(run)

    print("\nSummary:")
    print(json.dumps(run.get("metrics", {}), indent=4))
    print("\n✅ All updates complete.")
    print("Next: recompile the LaTeX manuscript to review formatting.")


def main():
    if ONE_SHOT:
        print("Fetching latest MLflow run...")
        try:
            run = fetch_latest_run()
        except ValueError as exc:
            print(f"Error fetching run: {exc}")
            return
        if run is None:
            print("No run found; exiting.")
            return
        process_run(run)
        return

    print("Entering watch loop; polling MLflow every 60 seconds.")
    while True:
        try:
            run = fetch_latest_run()
            if run:
                process_run(run)
        except ValueError as exc:
            print(f"Error fetching run: {exc}")
        time.sleep(60)

if __name__ == "__main__":
    main()
