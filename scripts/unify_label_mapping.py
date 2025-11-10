from __future__ import annotations
import ast
import io
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'dataset'
LOGS = ROOT / 'logs'
LOGS.mkdir(parents=True, exist_ok=True)

# Input paths (from requirements)
PTBXL_DB = DATASET / 'PTBXL' / 'ptbxl_database.csv'
PTBXL_SCP = DATASET / 'PTBXL' / 'scp_statements.csv'
CINC_REF = DATASET / 'CinC_2017_AFDB' / 'REFERENCE.csv'
PTBD_REC = DATASET / 'PTB_Diagnostic' / 'RECORDS'
DS6_SCT = DATASET / 'dataset6' / 'ConditionNames_SNOMED-CT.csv'
DS6_ROOT = DATASET / 'dataset6'

# Outputs
OUT_CSV = LOGS / 'unified_label_mapping.csv'
OUT_STATS = LOGS / 'label_stats.txt'

# Helpers

def safe_read_csv(path: Path, header: Optional[int] = 'infer', nrows: Optional[int] = None) -> pd.DataFrame:
    for enc in ('utf-8-sig', 'utf-8', 'latin-1'):
        try:
            if header == 'infer':
                df = pd.read_csv(path, sep=None, engine='python', encoding=enc, dtype=str, nrows=nrows)
            else:
                df = pd.read_csv(path, sep=None, engine='python', encoding=enc, dtype=str, nrows=nrows, header=header)
            return df
        except Exception:
            continue
    return pd.read_csv(path, dtype=str, nrows=nrows)


# Build PTBXL code -> meta rows map
PTBXL_META: Dict[str, Dict[str, str]] = {}
if PTBXL_SCP.exists():
    scp_df = safe_read_csv(PTBXL_SCP)
    # Normalize code index: first column is the code name; in PTBXL this is unnamed index column.
    # Ensure we have a 'code' column as index
    if 'Unnamed: 0' in scp_df.columns:
        scp_df = scp_df.rename(columns={'Unnamed: 0': 'code'})
    if 'code' not in scp_df.columns:
        # fallback: create from index
        scp_df = scp_df.rename_axis('code').reset_index()
    for _, r in scp_df.iterrows():
        code = str(r.get('code', '')).strip()
        if not code or code == 'nan':
            continue
        PTBXL_META[code] = {k: ('' if pd.isna(v) else str(v)) for k, v in r.items()}


def map_ptbxl_codes_to_class(codes: List[str]) -> str:
    # Priority: MI > AF > BBB > NORM > OTHER
    has_mi = False
    has_af = False
    has_bbb = False
    has_norm = False
    for c in codes:
        meta = PTBXL_META.get(c, {})
        diag_class = meta.get('diagnostic_class', '').upper()
        diag_sub = meta.get('diagnostic_subclass', '').upper()
        desc = meta.get('description', '').upper()
        rhythm_flag = str(meta.get('rhythm', '')).strip()
        # MI
        if diag_class == 'MI':
            has_mi = True
        # AF: check known AF codes or rhythm flag and description
        if c.upper() in {'AF', 'AFIB', 'AFIB1', 'AFL', 'AFLUT'} or 'ATRIAL FIBRILLATION' in desc:
            has_af = True
        elif rhythm_flag in ('1', 'True', 'true') and ('AF' in c.upper() or 'FIBR' in desc):
            has_af = True
        # BBB: conduction disturbances; look for subclass BBB or common codes
        if diag_class == 'CD' and ('BBB' in diag_sub or 'BBB' in desc or c.upper().endswith('BBB')):
            has_bbb = True
        # NORM
        if diag_class == 'NORM' or c.upper() == 'NORM':
            has_norm = True
    if has_mi:
        return 'MI'
    if has_af:
        return 'AF'
    if has_bbb:
        return 'BBB'
    if has_norm:
        return 'NORM'
    return 'OTHER'


def parse_scp_codes_field(raw: str) -> List[str]:
    if raw is None:
        return []
    s = str(raw)
    try:
        d = ast.literal_eval(s)
        if isinstance(d, dict):
            # use keys with weight > 0
            out = []
            for k, v in d.items():
                try:
                    if float(v) > 0:
                        out.append(str(k))
                except Exception:
                    out.append(str(k))
            return out
    except Exception:
        pass
    # Fallback: extract word-like codes from braces
    codes = re.findall(r"'([A-Za-z0-9_]+)'", s)
    if codes:
        return codes
    # Comma separated fallback
    return [t.strip() for t in s.split(',') if t.strip()]


def map_label(dataset: str, raw_label: str) -> str:
    ds = dataset.lower()
    if ds == 'ptbxl':
        # raw_label should be scp_codes field; map via scp_statements lookup
        codes = parse_scp_codes_field(raw_label)
        return map_ptbxl_codes_to_class(codes)
    if ds == 'cinc_2017_afdb':
        m = {'N': 'NORM', 'A': 'AF', 'O': 'OTHER'}
        return m.get(str(raw_label).strip(), 'OTHER')
    if ds == 'dataset6':
        # raw_label is a text possibly containing SNOMED codes; map key codes
        # 55827005->AF, 55930002->BBB, 233886008->MI
        s = str(raw_label)
        found = set(re.findall(r"\b(\d{5,9})\b", s))
        if '233886008' in found:
            return 'MI'
        if '55827005' in found:
            return 'AF'
        if '55930002' in found:
            return 'BBB'
        return 'OTHER'
    if ds == 'ptb_diagnostic':
        return 'MI'
    return 'OTHER'


def build_unified_dataframe() -> pd.DataFrame:
    rows: List[Tuple[str, str, str]] = []

    # PTBXL
    if PTBXL_DB.exists():
        df = safe_read_csv(PTBXL_DB)
        fname_col = 'filename_lr' if 'filename_lr' in df.columns else ('filename_hr' if 'filename_hr' in df.columns else None)
        if fname_col and 'scp_codes' in df.columns:
            for _, r in df[[fname_col, 'scp_codes']].dropna().iterrows():
                rid = str(r[fname_col]).strip()
                mapped = map_label('PTBXL', str(r['scp_codes']))
                rows.append(('PTBXL', rid, mapped))

    # CinC_2017_AFDB
    if CINC_REF.exists():
        df = safe_read_csv(CINC_REF, header=None)
        if df.shape[1] >= 2:
            df = df.rename(columns={0: 'record_id', 1: 'label'})
            for _, r in df[['record_id', 'label']].dropna().iterrows():
                rid = str(r['record_id']).strip()
                mapped = map_label('CinC_2017_AFDB', str(r['label']))
                rows.append(('CinC_2017_AFDB', rid, mapped))

    # dataset6: scan headers, but we also read mapping CSV for completeness (BOM-safe)
    if DS6_ROOT.exists():
        # Build label text per record from WFDB header comments
        for hea in DS6_ROOT.rglob('*.hea'):
            try:
                header = wfdb.rdheader(str(hea.with_suffix('')))
                comments = getattr(header, 'comments', None) or []
                raw = ' | '.join([str(c) for c in comments])
            except Exception:
                raw = ''
            rid = hea.relative_to(DATASET).with_suffix('').as_posix()
            mapped = map_label('dataset6', raw)
            rows.append(('dataset6', rid, mapped))

    # PTB_Diagnostic
    if PTBD_REC.exists():
        lines = PTBD_REC.read_text(encoding='utf-8', errors='replace').splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            rows.append(('PTB_Diagnostic', line, 'MI'))

    df = pd.DataFrame(rows, columns=['dataset', 'record_id', 'mapped_label'])
    return df


def save_outputs(df: pd.DataFrame) -> None:
    # Save unified CSV
    df.to_csv(OUT_CSV, index=False)
    # Save stats
    counts = df.groupby('mapped_label').size().sort_values(ascending=False)
    buf = io.StringIO()
    buf.write('Counts per mapped_label\n')
    buf.write('=' * 26 + '\n')
    for label, cnt in counts.items():
        buf.write(f'{label}: {int(cnt)}\n')
    OUT_STATS.write_text(buf.getvalue(), encoding='utf-8')


def main() -> int:
    df = build_unified_dataframe()
    save_outputs(df)
    print('Unified label mapping created successfully.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

