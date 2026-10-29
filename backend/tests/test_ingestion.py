
import pandas as pd
import pytest

from app.errors import APIError
from app.ingestion.reader import read_dataframe, write_parquet


def write(tmp_path, name, data: bytes):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def test_comma_semicolon_tab_and_pipe_delimiters(tmp_path, settings):
    for sep, ext in [(",", ".csv"), (";", ".csv"), ("\t", ".tsv"), ("|", ".csv")]:
        text = sep.join(["a", "b", "c"]) + "\n" + "\n".join(sep.join(map(str, [i, i * 2, "x"])) for i in range(30))
        df, rep = read_dataframe(write(tmp_path, f"f{ord(sep)}{ext}", text.encode()), ext, settings)
        assert list(df.columns) == ["a", "b", "c"] and len(df) == 30, sep


def test_bom_and_latin1(tmp_path, settings):
    df, _ = read_dataframe(write(tmp_path, "b.csv", b"\xef\xbb\xbfname,v\nA,1\nB,2\n"), ".csv", settings)
    assert list(df.columns) == ["name", "v"]
    df, rep = read_dataframe(write(tmp_path, "l.csv", "name,city\nA,Zürich\nB,Köln\n".encode("latin-1")), ".csv", settings)
    assert df["city"].tolist() == ["Zürich", "Köln"] and rep.warnings


def test_blank_and_duplicate_column_names_are_repaired(tmp_path, settings):
    df, rep = read_dataframe(write(tmp_path, "d.csv", b"a,a, ,b\n1,2,3,4\n"), ".csv", settings)
    assert list(df.columns) == ["a", "a_2", "unnamed_3", "b"] and rep.renamed_columns


def test_empty_and_header_only_files_rejected(tmp_path, settings):
    with pytest.raises(APIError) as e:
        read_dataframe(write(tmp_path, "e.csv", b""), ".csv", settings)
    assert e.value.status == 422
    with pytest.raises(APIError):
        read_dataframe(write(tmp_path, "h.csv", b"a,b,c\n"), ".csv", settings)


def test_parquet_roundtrip_and_mixed_object_column(tmp_path, settings):
    df = pd.DataFrame({"a": [1, 2, 3], "mixed": [1, "x", 2.5]})
    p = str(tmp_path / "x.parquet")
    write_parquet(df, p)
    back, _ = read_dataframe(p, ".parquet", settings)
    assert len(back) == 3


def test_excel_and_sheet_selection(tmp_path, settings):
    p = str(tmp_path / "x.xlsx")
    with pd.ExcelWriter(p) as w:
        pd.DataFrame({"a": [1, 2]}).to_excel(w, sheet_name="first", index=False)
        pd.DataFrame({"b": [3, 4, 5]}).to_excel(w, sheet_name="second", index=False)
    df, rep = read_dataframe(p, ".xlsx", settings)
    assert list(df.columns) == ["a"] and rep.sheets_available == ["first", "second"]
    df, _ = read_dataframe(p, ".xlsx", settings, sheet="second")
    assert len(df) == 3
    with pytest.raises(APIError):
        read_dataframe(p, ".xlsx", settings, sheet="nope")
