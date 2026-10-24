from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .drawing import check_pdf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chk-dwg-rev",
        description="Extract drawing revision letters from engineering-drawing PDFs.",
    )
    parser.add_argument("input_dir", type=Path, help="directory containing PDF files")
    parser.add_argument("output_file", type=Path, help="file to write the results to")
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="search the input directory recursively",
    )
    return parser


def find_pdfs(input_dir: Path, recursive: bool) -> list[Path]:
    paths = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(
        path
        for path in paths
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.input_dir.is_dir():
        parser.error(f"input directory does not exist: {args.input_dir}")

    pdf_files = find_pdfs(args.input_dir, args.recursive)
    if not pdf_files:
        print(f"warning: no PDF files found in {args.input_dir}", file=sys.stderr)

    with args.output_file.open("w", encoding="utf-8") as output:
        for pdf_path in pdf_files:
            try:
                is_drawing, revision = check_pdf(pdf_path)
            except Exception as exc:  # noqa: BLE001 - keep processing other files
                print(f"warning: could not process {pdf_path}: {exc}", file=sys.stderr)
                continue

            if not is_drawing:
                print(f"skipping (not an engineering drawing): {pdf_path}", file=sys.stderr)
                continue

            if revision is None:
                print(f"warning: no REV box found in {pdf_path}", file=sys.stderr)
                continue

            line = f"{pdf_path}, {revision}"
            output.write(f"{line}\n")
            output.flush()
            print(line)

    return 0
