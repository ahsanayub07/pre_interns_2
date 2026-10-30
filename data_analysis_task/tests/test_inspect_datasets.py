import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import inspect_datasets as ids


def test_scan_csv_counts_rows_blanks_and_ids(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("id,title,body\n1,A,\n2,,text\n3,C,text\n", encoding="utf-8")
    r = ids.scan_csv(str(p), "id")
    assert r["rows"] == 3 and r["ids"] == ["1", "2", "3"]
    assert r["blank_by_column"] == {"id": 0, "title": 1, "body": 1}


def test_scan_csv_reads_bom_and_long_fields(tmp_path):
    p = tmp_path / "b.csv"
    p.write_text("﻿id,body\n1," + "x" * 300000 + "\n", encoding="utf-8")
    r = ids.scan_csv(str(p), "id")
    assert r["columns"] == ["id", "body"] and r["rows"] == 1


def test_is_blank():
    assert ids.is_blank(None) and ids.is_blank("  ") and not ids.is_blank("a")
