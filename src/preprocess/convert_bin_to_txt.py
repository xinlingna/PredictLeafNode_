#!/usr/bin/env python3
"""Convert a raw (headerless) binary matrix file (.bin) to a whitespace-delimited text file (.txt).

This repo already uses *raw binary arrays* in multiple places (e.g. `np.fromfile` with a known shape).
This script is a lightweight helper to export such arrays as text.

Assumptions (default):
- The binary file contains a contiguous array with shape [N, D]
- No header; row-major layout
- dtype defaults to float32

Example (your case):
python -m src.preprocess.convert_bin_to_txt \
  --input_bin /home/xln/PredictLeafNode/input/Training_data/sift10M/sift10M_query.bin \
  --output_txt /home/xln/PredictLeafNode/input/Training_data/sift10M/sift10M_query.txt \
  --d 128

If you know N explicitly you can pass `--n`.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np


def _parse_dtype(dtype_str: str) -> np.dtype:
    try:
        return np.dtype(dtype_str)
    except TypeError as e:
        raise ValueError(f"Unsupported dtype: {dtype_str}") from e


def infer_n(file_path: str, d: int, dtype: np.dtype) -> int:
    if d <= 0:
        raise ValueError(f"d must be > 0, got {d}")

    file_size = os.path.getsize(file_path)
    itemsize = int(dtype.itemsize)
    row_bytes = d * itemsize
    if row_bytes <= 0:
        raise ValueError("Invalid row byte size")

    if file_size % row_bytes != 0:
        raise ValueError(
            f"File size ({file_size} bytes) is not divisible by (d * itemsize) = {row_bytes}. "
            f"Check --d/--dtype or whether the file has a header."
        )

    return file_size // row_bytes


def convert_bin_to_txt(
    input_bin: str,
    output_txt: str,
    d: int,
    dtype: np.dtype,
    n: Optional[int],
    fmt: str,
    delimiter: str,
    chunk_rows: int,
) -> None:
    if n is None:
        n = infer_n(input_bin, d=d, dtype=dtype)

    expected_elems = n * d

    os.makedirs(os.path.dirname(os.path.abspath(output_txt)) or ".", exist_ok=True)

    # Chunked conversion to avoid large peak RAM for huge datasets.
    # We stream from disk -> numpy array chunk -> savetxt.
    with open(input_bin, "rb") as fin, open(output_txt, "w", encoding="utf-8") as fout:
        rows_written = 0
        while rows_written < n:
            rows_to_read = min(chunk_rows, n - rows_written)
            elems_to_read = rows_to_read * d

            chunk = np.fromfile(fin, dtype=dtype, count=elems_to_read)
            if chunk.size != elems_to_read:
                raise ValueError(
                    f"Unexpected EOF while reading {input_bin}. "
                    f"Expected {expected_elems} elements total, got only {rows_written * d + chunk.size}."
                )

            chunk = chunk.reshape(rows_to_read, d)
            np.savetxt(fout, chunk, fmt=fmt, delimiter=delimiter)
            rows_written += rows_to_read

        # Ensure there is no trailing unread data.
        tail = fin.read(1)
        if tail not in (b"",):
            raise ValueError(
                "Input file has extra bytes beyond n*d elements. "
                "Remove --n or re-check --d/--dtype."
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert raw binary matrix (.bin) to whitespace-delimited text (.txt)")
    ap.add_argument("--input_bin", required=True, help="Input .bin file (raw, headerless)")
    ap.add_argument("--output_txt", required=True, help="Output .txt file")
    ap.add_argument("--d", type=int, required=True, help="Vector dimension D")
    ap.add_argument("--n", type=int, default=None, help="Number of rows N (optional; inferred from file size if omitted)")
    ap.add_argument(
        "--dtype",
        default="float32",
        help="Numpy dtype of stored elements (default: float32). Examples: float32, uint8, int32",
    )
    ap.add_argument("--fmt", default="%.6g", help="Output number format for np.savetxt (default: %%.6g)")
    ap.add_argument("--delimiter", default=" ", help="Delimiter between columns (default: space)")
    ap.add_argument(
        "--chunk_rows",
        type=int,
        default=10000,
        help="Rows per chunk when writing (default: 10000). Increase for speed, decrease for RAM.",
    )

    args = ap.parse_args()

    dtype = _parse_dtype(args.dtype)

    convert_bin_to_txt(
        input_bin=args.input_bin,
        output_txt=args.output_txt,
        d=args.d,
        dtype=dtype,
        n=args.n,
        fmt=args.fmt,
        delimiter=args.delimiter,
        chunk_rows=args.chunk_rows,
    )


if __name__ == "__main__":
    main()
