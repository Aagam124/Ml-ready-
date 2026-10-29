"""Read CSV / TSV / Excel / Parquet into a DataFrame and write Parquet back out."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from app.config import Settings
from app.errors import APIError


@dataclass
class ReadReport:
    encoding: Optional[str] = None
    delimiter: Optional[str] = None
    sheet: Optional[str] = None
    sheets_available: List[str] = field(default_factory=list)
    renamed_columns: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _sniff_delimiter(head_text: str) -> str:
    try:
        return csv.Sniffer().sniff(head_text, delimiters=",;\t|").delimiter
    except csv.Error:
        lines = [ln for ln in head_text.splitlines() if ln.strip()][:20]
        best, best_score = ",", 0
        for d in [",", ";", "\t", "|"]:
            counts = [ln.count(d) for ln in lines]
            if counts and min(counts) > 0:
                score = min(counts) * (2 if len(set(counts)) == 1 else 1)
                if score > best_score:
                    best, best_score = d, score
        return best


def _clean_names(raw_names, report: ReadReport) -> List[str]:
    """Strip, fill blanks, and de-duplicate column names, recording every change."""
    seen: dict = {}
    new_cols = []
    for i, raw in enumerate(raw_names):
        raw_s = "" if raw is None or (isinstance(raw, float) and raw != raw) else str(raw)
        name = raw_s.strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"unnamed_{i + 1}"
        base, n = name, 1
        while name in seen:
            n += 1
            name = f"{base}_{n}"
        seen[name] = True
        if name != raw_s:
            report.renamed_columns.append({"from": raw_s, "to": name})
        new_cols.append(name)
    return new_cols


def _normalise_columns(df: pd.DataFrame, report: ReadReport) -> pd.DataFrame:
    df.columns = _clean_names(list(df.columns), report)
    return df


def _read_csv(path: str, ext: str, report: ReadReport) -> pd.DataFrame:
    with open(path, "rb") as f:
        head = f.read(64 * 1024)
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            head_text = head.decode(enc, errors="strict" if enc != "latin-1" else "replace")
            break
        except UnicodeDecodeError:
            # A multi-byte character may be cut at the 64 KB boundary; retry trimmed.
            try:
                head_text = head[:-4].decode(enc)
                break
            except UnicodeDecodeError:
                continue
    delimiter = "\t" if ext == ".tsv" else _sniff_delimiter(head_text)
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            raw = pd.read_csv(path, sep=delimiter, encoding=enc, header=None, nrows=1, dtype=str)
            names = _clean_names(list(raw.iloc[0]), report)
            df = pd.read_csv(path, sep=delimiter, encoding=enc, low_memory=False,
                             skip_blank_lines=True, header=0, names=names)
            report.encoding, report.delimiter = enc, delimiter
            if enc != "utf-8-sig":
                report.warnings.append(f"File is not UTF-8; decoded as {enc}.")
            return df
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except pd.errors.EmptyDataError:
            raise APIError(422, "empty_file", "The file contains no data.")
        except pd.errors.ParserError as e:
            raise APIError(422, "parse_error", f"Could not parse the file as delimited text: {e}")
    raise APIError(422, "encoding_error", f"Could not decode the file: {last_err}")


def read_dataframe(path: str, ext: str, settings: Settings, sheet: Optional[str] = None):
    """Returns (DataFrame, ReadReport). Raises APIError(422) for unreadable/invalid data."""
    report = ReadReport()
    try:
        if ext in (".csv", ".tsv", ".txt"):
            df = _read_csv(path, ext, report)
        elif ext in (".xlsx", ".xls"):
            engine = "openpyxl" if ext == ".xlsx" else "xlrd"
            with pd.ExcelFile(path, engine=engine) as xl:
                report.sheets_available = list(xl.sheet_names)
                chosen = sheet if sheet else xl.sheet_names[0]
                if chosen not in xl.sheet_names:
                    raise APIError(422, "sheet_not_found", f"Sheet '{chosen}' not found.",
                                   {"available": xl.sheet_names})
                report.sheet = chosen
                raw = xl.parse(chosen, header=None, nrows=1, dtype=str)
                names = _clean_names(list(raw.iloc[0]), report)
                df = xl.parse(chosen, header=0, names=names)
            if len(report.sheets_available) > 1:
                report.warnings.append(f"Workbook has {len(report.sheets_available)} sheets; used '{chosen}'.")
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise APIError(422, "unsupported_file_type", f"Unsupported file type '{ext}'.")
    except APIError:
        raise
    except Exception as e:  # corrupt workbook, bad parquet, etc.
        raise APIError(422, "unreadable_file", f"Could not read the file: {type(e).__name__}: {e}")

    df = _normalise_columns(df, report)
    if df.shape[1] == 0 or df.shape[0] == 0:
        raise APIError(422, "empty_dataset", "The file has no data rows or no columns.")
    if df.shape[1] > settings.max_columns:
        raise APIError(422, "too_many_columns", f"Dataset has {df.shape[1]} columns; the limit is {settings.max_columns}.")
    return df, report


def write_parquet(df: pd.DataFrame, path: str) -> None:
    """Write Parquet; if mixed-type object columns break Arrow, stringify just those."""
    try:
        df.to_parquet(path, index=False)
        return
    except Exception:
        fixed = df.copy()
        for c in fixed.columns:
            if fixed[c].dtype == object:
                fixed[c] = fixed[c].map(lambda v: v if v is None or v != v else str(v))
        fixed.to_parquet(path, index=False)
