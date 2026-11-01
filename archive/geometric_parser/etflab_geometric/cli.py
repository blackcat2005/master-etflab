"""CLI Phase 0: PDF -> JSON (bang + van ban). Chay bang ``python -m etflab``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import IngestConfig, process_pdf

DEFAULT_RAW = Path("data/raw")
DEFAULT_OUT = Path("data/processed")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def iter_pdfs(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if not target.exists():
        return []
    return sorted(p for p in target.rglob("*.pdf") if p.is_file())


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="etflab",
        description=(
            "Phase 0 ingestion: PDF -> JSON. Mac dinh xu ly ca data/raw/."
        ),
    )
    ap.add_argument(
        "target",
        nargs="?",
        type=Path,
        default=DEFAULT_RAW,
        help=f"file PDF hoac thu muc chua PDF (mac dinh: {DEFAULT_RAW})",
    )
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT,
                    help=f"thu muc ghi JSON (mac dinh: {DEFAULT_OUT})")
    ap.add_argument("--indent", type=int, default=2, help="indent JSON (mac dinh: 2)")
    ap.add_argument("--quiet", action="store_true", help="chi in loi + tong ket")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pdfs = iter_pdfs(args.target)
    if not pdfs:
        print(f"[!] khong tim thay PDF nao trong {args.target}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    cfg = IngestConfig()
    total_tables = 0
    n_failed = 0
    n_warning = 0

    for pdf in pdfs:
        try:
            result = process_pdf(pdf, cfg)
        except Exception as exc:  # noqa: BLE001 - CLI khong duoc dung mot PDF lam dung ca lo chay
            print(f"[x] {pdf}: {type(exc).__name__}: {exc}", file=sys.stderr)
            n_failed += 1
            continue

        dest = args.out / (pdf.stem + ".json")
        dest.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=args.indent),
            encoding="utf-8",
        )
        total_tables += len(result.tables)
        n_warning += len(result.warnings)
        if not args.quiet:
            prose = sum(p.n_blocks for p in result.pages_text)
            print(f"[+] {pdf.relative_to(Path.cwd()) if pdf.is_absolute() else pdf}: "
                  f"{result.n_pages} trang, {len(result.tables)} bang, {prose} khoi van ban")
            for w in result.warnings:
                print(f"      [!] {w}")
            for t in result.tables:
                print(f"      {t.table_id}  p{t.page_span}  {t.n_rows}x{t.n_cols}")
                print(f"        cols={t.columns}")

    print(f"\nTong: {len(pdfs)} file, {total_tables} bang, {n_warning} canh bao"
          + (f", {n_failed} that bai" if n_failed else "") + ".")
    return 0 if not n_failed else 2


if __name__ == "__main__":
    raise SystemExit(main())