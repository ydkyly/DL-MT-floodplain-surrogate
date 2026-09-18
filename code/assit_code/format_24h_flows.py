# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pathlib import Path

def _ensure_dir(path):
    """Ensure the directory exists. If not, create it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _to_24_list(x):
    """Convert a sequence to a list of 24 valid numeric values."""
    s = pd.Series(np.array(x).reshape(-1)).astype(float)
    if len(s) < 24:
        raise ValueError(f"Sequence has only {len(s)} valid numeric values; need >=24.")
    return s.iloc[:24].tolist()

def read_24h_columns_from_excel(xlsx_path):
    """Read 24-hour sequences from an Excel file."""
    # Load the Excel file
    df = pd.read_excel(xlsx_path, sheet_name=0, header=0)
    print("jiazaid ", df.head())  # Debug: Check the first few rows of the DataFrame

    sequences = []
    for col in df.columns:
        # Directly convert to numeric values and ensure 24 values are present
        series = pd.to_numeric(df[col], errors="coerce")  # Convert to numeric, coercing errors to NaN
        if len(series) >= 24:
            sequences.append(series.iloc[:24].astype(float).tolist())  # Only take the first 24 valid numbers
        else:
            print(f"Warning: Column {col} does not have enough valid values (24 required).")
    return sequences

def write_sequences_to_excels(sequences, out_dir, start_index=1, zfill=None):
    """Write each sequence to its own Excel file."""
    out_dir = _ensure_dir(out_dir)
    seqs = []

    for s in sequences:
        seqs.append(_to_24_list(s))

    if zfill is None:
        last_index = start_index + max(len(seqs) - 1, 0)
        zfill = max(2, len(str(last_index)))

    written = []
    for i, seq in enumerate(seqs, start=start_index):
        fname = f"{str(i).zfill(zfill)}.xlsx"
        fpath = out_dir / fname
        k = 1
        final_path = fpath
        while final_path.exists():
            final_path = out_dir / f"{str(i).zfill(zfill)}_{k}.xlsx"
            k += 1

        # Create the DataFrame for output
        df_out = pd.DataFrame({"Hour": list(range(24)), "Q": seq})
        with pd.ExcelWriter(final_path, engine="openpyxl") as writer:
            df_out.to_excel(writer, index=False, sheet_name="Flow24h")
        written.append(final_path)

    return written

def process_inputs(excel_paths=None, sequences=None, out_dir="flows_output", start_index=1, zfill=None):
    """Process the input sequences and save them to Excel files."""
    all_seqs = []

    # Check and process input Excel paths
    if excel_paths:
        for path in excel_paths:
            sequences_from_excel = read_24h_columns_from_excel(path)
            all_seqs.extend(sequences_from_excel)

    # If sequences are passed directly
    if sequences:
        for s in sequences:
            all_seqs.append(_to_24_list(s))

    if not all_seqs:
        raise ValueError("No valid sequences found. Provide sequences and/or excel_paths.")

    written = write_sequences_to_excels(all_seqs, out_dir, start_index=start_index, zfill=zfill)
    return written

# Example usage:
# Update the following paths before running
out_dir = "D:/Work/qyb/ResNet-18/data/train/inflow"  # Modify output directory
excel_paths = ["D:/Work/qyb/ResNet-18/new.xlsx"]  # Excel file path(s)

# Process the inputs and write the output files
written_files = process_inputs(excel_paths=excel_paths, out_dir=out_dir, start_index=1)

print(f"Files written: {len(written_files)}")
