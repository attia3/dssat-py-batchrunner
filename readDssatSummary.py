# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:48:39 2025

@author: ahmed.attia
"""

import pandas as pd
import io
import re


filename = "C:/DSSAT48/Sequence/Summary.OUT"

def read_dssat_summary(filename):
    """ Reads the DSSAT Summary.OUT file as a single DataFrame """
    
    with open(filename, "r", encoding="cp437") as f:
        lines = f.readlines()

 # Locate the header row (starting with @)
    header_index = next(i for i, line in enumerate(lines) if line.strip().startswith("@"))
    
    # Extract column names, removing '@'
    columns = re.split(r'\s+', lines[header_index].strip())[1:]  # Ignore "@"

    # Read data section, ensuring multi-word treatments stay together
    data_lines = lines[header_index + 1:]
    data = []
    
    for line in data_lines:
        if line.strip():
            # Split first 8 columns (before treatment name)
            first_columns = re.split(r'\s+', line.strip(), maxsplit=8)  # First 8 columns are safe
            remaining_line = first_columns.pop()  # The remaining part contains potential multi-word text

            # Identify the treatment name (e.g., "Winter wheat") and extract remaining columns correctly
            match = re.match(r"(\D+)\s+(.*)", remaining_line)  # Match first non-digit part as treatment name
            if match:
                treatment_name = match.group(1).strip()  # Extract full treatment name
                rest_of_columns = re.split(r'\s+', match.group(2).strip())  # Remaining numeric columns
            else:
                treatment_name = remaining_line.strip()
                rest_of_columns = []

            # Combine correctly
            row = first_columns + [treatment_name] + rest_of_columns
            data.append(row)

    # Check final column alignment
    max_columns = max(len(row) for row in data)
    
    # If missing columns, add placeholders
    for row in data:
        if len(row) < max_columns:
            row.extend(["-99"] * (max_columns - len(row)))

    # Ensure column count matches
    if len(columns) < max_columns:
        extra_cols = [f"COL_{i}" for i in range(len(columns), max_columns)]
        columns.extend(extra_cols)

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Convert numerical columns where applicable
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])  # Convert to numeric if possible
        except ValueError:
            pass  # Ignore errors for non-numeric columns

    return df

# Convert DSSAT-style Julian dates to standard format
def convert_dssat_date(julian_date):
    """Convert DSSAT-style Julian dates (YYYYDDD) to YYYY-MM-DD format"""
    try:
        year = int(str(julian_date)[:4])
        day_of_year = int(str(julian_date)[4:])
        return pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(day_of_year - 1, unit="D")
    except:
        return pd.NaT  # Return NaT for missing values


