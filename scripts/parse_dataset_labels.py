from __future__ import annotations
import re
import ast
import io
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'dataset'
LOGS = ROOT / 'logs'
LOGS.mkdir(parents=True, exist_ok=True)

OUT_CSV = LOGS / 'all_dataset_labels.csv'
OUT_SUMMARY = LOGS / 'label_extraction_summary.txt'

# Utilities

def safe_read_csv(path: Path, nrows: int | None = None, expect_header: bool | None = None) -> pd.DataFrame:
    # Try encodings with engine=python and sep inference. Handle BOM via utf-8-sig.
    for enc in ('utf-8-sig', 'utf-8', 'latin-1'):
        try:
            if expect_header is None:
                df = pd.read_csv(path, nrows=nrows, sep=None, engine='python', encoding=enc, dtype=str)
            elif expect_header:
                df = pd.read_csv(path, nrows=nrows, sep=None, engine='python', encoding=enc, dtype=str, header=0)
            else:
                df = pd.read_csv(path, nrows=nrows, sep=None, engine='python', encoding=enc, dtype=str, header=None)
            return df
        except Exception:
            continue
    # Final fallback
    return pd.read_csv(path, nrows=nrows, dtype=str)


def normalize_label_list(labels: List[str]) -> str:
    # Join multi-labels with '|', drop empties, deduplicate preserving order
    seen = set()
    out = []
    for l in labels:
        if not l:
            continue
        if l not in seen:
            seen.add(l)
            out.append(l)
    return '|'.join(out) if out else ''


# PTBXL: ptbxl_database.csv (scp_codes -> keys with value>0)

def extract_ptbxl() -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    ds_dir = DATASET / 'PTBXL'
    csv_path = ds_dir / 'ptbxl_database.csv'
    if not csv_path.exists():
        return rows
    df = safe_read_csv(csv_path)
    # Columns may include filename_hr or filename_lr
    fname_col = 'filename_hr' if 'filename_hr' in df.columns else ('filename_lr' if 'filename_lr' in df.columns else None)
    scp_col = 'scp_codes' if 'scp_codes' in df.columns else None
    if not fname_col or not scp_col:
        return rows
    for _, r in df[[fname_col, scp_col]].dropna().iterrows():
        record_id = str(r[fname_col]).strip()
        raw = str(r[scp_col]).strip()
        labels: List[str] = []
        try:
            d = ast.literal_eval(raw)
            if isinstance(d, dict):
                for k, v in d.items():
                    try:
                        if float(v) > 0:
                            labels.append(str(k))
                    except Exception:
                        labels.append(str(k))
            else:
                # fallback: comma-separated
                labels = [s.strip() for s in raw.split(',') if s.strip()]
        except Exception:
            # fallback: parse like { 'CODE': number }
            labels = [s.strip() for s in re.findall(r"'([A-Z0-9_]+)'\s*:\s*[-0-9.]+", raw)]
        label = normalize_label_list(labels)
        rows.append(('PTBXL', record_id, label))
    return rows


# PTB_Diagnostic: RECORDS text, label = patient folder name (first path segment)

def extract_ptb_diagnostic() -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    ds_dir = DATASET / 'PTB_Diagnostic'
    rec_file = ds_dir / 'RECORDS'
    if not rec_file.exists():
        return rows
    content = rec_file.read_text(encoding='utf-8', errors='replace').splitlines()
    for line in content:
        line = line.strip()
        if not line:
            continue
        parts = line.split('/')
        label = parts[0] if parts else ''
        rows.append(('PTB_Diagnostic', line, label))
    return rows


# CinC_2017_AFDB: REFERENCE*.csv (two columns: id,label), prefer REFERENCE.csv then v3..v0

def extract_cinc_afdb() -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    ds_dir = DATASET / 'CinC_2017_AFDB'
    if not ds_dir.exists():
        return rows
    candidates = []
    # top-level and nested
    for p in ds_dir.rglob('REFERENCE*.csv'):
        candidates.append(p)
    # Prioritize preference
    priority_order = {
        'REFERENCE.csv': 0,
        'REFERENCE-v3.csv': 1,
        'REFERENCE-v2.csv': 2,
        'REFERENCE-v1.csv': 3,
        'REFERENCE-v0.csv': 4
    }
    candidates.sort(key=lambda p: (priority_order.get(p.name, 99), str(p)))

    seen_ids: set[str] = set()
    for csv_path in candidates:
        # Try reading without header; many files have two columns of values directly
        df = safe_read_csv(csv_path, expect_header=False)
        if df.shape[1] < 2:
            # try with header
            df = safe_read_csv(csv_path, expect_header=True)
        if df.shape[1] < 2:
            continue
        # Use first two columns as id and label
        id_col = df.columns[0]
        label_col = df.columns[1]
        for _, r in df[[id_col, label_col]].dropna().iterrows():
            rid = str(r[id_col]).strip()
            lab = str(r[label_col]).strip()
            # Skip header-looking rows
            if rid.lower().startswith('a00') or rid.lower().startswith('a0') or '/' in rid:
                pass
            # For validation/training subdirs, sometimes header names are already record ids in first row; we accept all
            if rid not in seen_ids:
                seen_ids.add(rid)
                rows.append(('CinC_2017_AFDB', rid, lab))
    return rows


# dataset6: map SNOMED codes in WFDB headers to names using ConditionNames_SNOMED-CT.csv

SNOMED_PATTERN = re.compile(r"\b(\d{5,9})\b")

def load_snomed_mapping(ds6_root: Path) -> Dict[str, str]:
    map_path = ds6_root / 'ConditionNames_SNOMED-CT.csv'
    mapping: Dict[str, str] = {}
    if not map_path.exists():
        return mapping
    df = safe_read_csv(map_path)
    # Normalize column names for BOM and spaces
    cols = {c.strip().lstrip('\ufeff'): c for c in df.columns}
    snomed_col = None
    for cand in ('Snomed_CT', 'SNOMED_CT', 'Snomed', 'SNOMED'):
        if cand in cols:
            snomed_col = cols[cand]
            break
    name_col = None
    for cand in ('Acronym Name', 'Acronym', 'Name', 'Full Name'):
        if cand in cols:
            name_col = cols[cand]
            break
    if not snomed_col:
        return mapping
    if not name_col:
        # fallback to first other column
        name_col = [c for c in df.columns if c != snomed_col][0]
    for _, r in df[[snomed_col, name_col]].dropna().iterrows():
        code = str(r[snomed_col]).strip()
        name = str(r[name_col]).strip()
        if code and name:
            mapping[code] = name
    return mapping


def extract_dataset6() -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    ds_dir = DATASET / 'dataset6'
    if not ds_dir.exists():
        return rows
    snomed_map = load_snomed_mapping(ds_dir)
    # Iterate all .hea headers under WFDBRecords
    for hea in ds_dir.rglob('*.hea'):
        try:
            header = wfdb.rdheader(str(hea.with_suffix('')))
            comments = getattr(header, 'comments', None) or []
            text = ' | '.join(comments)
            codes = SNOMED_PATTERN.findall(text)
            labels: List[str] = []
            for c in codes:
                labels.append(snomed_map.get(c, c))
            label = normalize_label_list(labels)
            # record_id as relative path without extension
            rid = hea.relative_to(DATASET).with_suffix('').as_posix()
            rows.append(('dataset6', rid, label))
        except Exception:
            # Fallback: use file stem as id and empty label
            rid = hea.relative_to(DATASET).with_suffix('').as_posix()
            rows.append(('dataset6', rid, ''))
    return rows


def main() -> int:
    all_rows: List[Tuple[str, str, str]] = []

    # Extract from sources
    all_rows.extend(extract_ptbxl())
    all_rows.extend(extract_ptb_diagnostic())
    all_rows.extend(extract_cinc_afdb())
    all_rows.extend(extract_dataset6())

    if not all_rows:
        print('No labels extracted. Nothing to write.')
        return 0

    df = pd.DataFrame(all_rows, columns=['dataset', 'record_id', 'label'])
    # Save consolidated CSV
    df.to_csv(OUT_CSV, index=False)

    # Build summary
    buf = io.StringIO()
    buf.write('Label Extraction Summary\n')
    buf.write('=' * 30 + '\n\n')
    # Row counts per dataset
    counts = df.groupby('dataset').size().sort_values(ascending=False)
    buf.write('Rows per dataset:\n')
    for ds, cnt in counts.items():
        buf.write(f'- {ds}: {int(cnt)}\n')
    buf.write('\n')

    # Top 10 labels across datasets (split multi-label by '|')
    tokens: List[str] = []
    for s in df['label'].astype(str):
        parts = [p for p in (s.split('|') if s else []) if p]
        tokens.extend(parts)
    if tokens:
        top = pd.Series(tokens).value_counts().head(10)
        buf.write('Top 10 labels:\n')
        for name, cnt in top.items():
            buf.write(f'- {name}: {int(cnt)}\n')
        buf.write('\n')
    else:
        buf.write('Top 10 labels: (no labels found)\n\n')

    OUT_SUMMARY.write_text(buf.getvalue(), encoding='utf-8')
    print(f'Wrote consolidated labels to {OUT_CSV.as_posix()}')
    print(f'Wrote summary to {OUT_SUMMARY.as_posix()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

