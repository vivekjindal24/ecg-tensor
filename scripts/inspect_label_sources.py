from __future__ import annotations
import sys
from pathlib import Path
import io
import json
import traceback

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / 'dataset'
LOGS_DIR = ROOT / 'logs'
OUT_FILE = LOGS_DIR / 'dataset_label_sources_summary.txt'

# Datasets to scan (relative to DATASET_DIR)
TARGET_DATASETS = [
    'PTBXL',
    'PTB_Diagnostic',
    'CinC_2017_AFDB',
    'Chapman_Shaoxing',
    'dataset4',
    'dataset5',
    'dataset6',
]

# File name patterns to consider
CSV_SUFFIX = '.csv'
INTERESTING_TEXT_FILES = {
    'RECORDS.txt',
    'RECORDS',
}

LIKELY_LABEL_KEYS = {
    'diagnosis', 'diagnoses', 'diagnostic', 'scp_codes', 'label', 'labels',
    'class', 'classes', 'rhythm', 'dx', 'Dx', 'Dx1', 'Dx2', 'Dx3', 'Dx4'
}


def is_interesting_csv(p: Path) -> bool:
    name = p.name.lower()
    if not name.endswith(CSV_SUFFIX):
        return False
    # common metadata names
    keywords = [
        'database', 'annotation', 'labels', 'label', 'meta', 'metadata',
        'classes', 'diagn', 'scp', 'info'
    ]
    return any(k in name for k in keywords) or True  # accept all CSV, filter later by content


def safe_read_csv_head(path: Path, nrows: int = 5) -> pd.DataFrame:
    # Try UTF-8 with heuristic sep detection; fallback to latin-1
    for enc in ('utf-8', 'latin-1'):
        try:
            df = pd.read_csv(path, nrows=nrows, sep=None, engine='python', encoding=enc, dtype=str)
            return df
        except Exception:
            continue
    # Last resort: let pandas guess with defaults
    return pd.read_csv(path, nrows=nrows, dtype=str)


def summarize_dataframe(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    # Column names
    cols = list(df.columns)
    buf.write(f"Columns ({len(cols)}): {cols}\n")
    # Head
    head_str = df.head(5).to_string(index=False)
    buf.write("Sample (head):\n")
    buf.write(head_str + "\n")
    # Detect likely label columns
    lower_cols = [c.lower() for c in cols]
    matches = [c for c in cols if c.lower() in {k.lower() for k in LIKELY_LABEL_KEYS}]
    if not matches:
        # try partial contains
        for c in cols:
            lc = c.lower()
            if any(k in lc for k in ['diagn', 'scp', 'label', 'class', 'rhythm', 'dx']):
                matches.append(c)
    if matches:
        buf.write(f"Likely label columns: {matches}\n")
    else:
        buf.write("Likely label columns: (none detected)\n")
    return buf.getvalue()


def summarize_text_file(path: Path, nlines: int = 5) -> str:
    try:
        content = path.read_text(encoding='utf-8', errors='replace').splitlines()
    except Exception:
        content = path.read_text(errors='replace').splitlines()
    sample = "\n".join(content[:nlines])
    return f"First {nlines} line(s):\n{sample}\n"


def main() -> int:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report = io.StringIO()
    report.write("ECG Dataset Label Sources Summary\n")
    report.write("=" * 40 + "\n\n")

    for ds in TARGET_DATASETS:
        ds_path = DATASET_DIR / ds
        report.write(f"Dataset: {ds}\n")
        report.write(f"Path: {ds_path.as_posix()}\n")
        if not ds_path.exists():
            report.write("Status: MISSING (skipped)\n\n")
            continue

        # Collect candidate files
        files: list[Path] = []
        for p in ds_path.rglob('*'):
            if p.is_file():
                if p.name in INTERESTING_TEXT_FILES:
                    files.append(p)
                elif p.suffix.lower() == CSV_SUFFIX and is_interesting_csv(p):
                    files.append(p)
        if not files:
            report.write("No CSV or known metadata text files found.\n\n")
            continue

        for f in sorted(files):
            report.write(f"-- File: {f.relative_to(ROOT).as_posix()}\n")
            try:
                if f.suffix.lower() == CSV_SUFFIX:
                    df = safe_read_csv_head(f, nrows=5)
                    report.write("Type: CSV\n")
                    report.write(summarize_dataframe(df))
                else:
                    report.write("Type: TEXT\n")
                    report.write(summarize_text_file(f, nlines=5))
            except Exception as e:
                report.write(f"Error reading file: {e}\n")
                report.write(traceback.format_exc() + "\n")
            report.write("\n")
        report.write("\n")

    OUT_FILE.write_text(report.getvalue(), encoding='utf-8')
    print(f"Multi-dataset label summary saved to {OUT_FILE.as_posix()}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

