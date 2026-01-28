# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:26:57 2025

@author: ahmed.attia
"""
import os
import xarray as xr
import pandas as pd
import numpy as np
import re

# Atmospheric pressure in hPa (assumed constant for simplicity)
P = 1013.25  # Standard atmospheric pressure in hPa

# Mapping of expected variable names to actual variable names in the dataset
variable_mapping = {
    "rsds": "surface_downwelling_shortwave_flux_in_air",
    "tasmax": "air_temperature",
    "tasmin": "air_temperature",
    "pr": "precipitation",
    "was": "wind_speed",
    "huss": "specific_humidity"
}

def convert_units(variable, values):
    """Convert units to match DSSAT requirements."""
    if variable == "rsds":  # W/m² to MJ/m²/day
        return values * 0.0864
    elif variable in ["tasmax", "tasmin"]:  # K to °C
        return values - 273.15
    elif variable == "pr":  # Already in mm/day, no conversion needed
        return values
    elif variable == "was":  # m/s to km/day
        return values * 86.4
    elif variable == "huss": 
        return values  # Specific humidity, no conversion here
    return values

# Function to adjust longitudes if they are in 0-360 range
def adjust_longitudes(data_array):
    """Adjust longitudes in an xarray DataArray to -180 to 180."""
    return xr.where(data_array > 180, data_array - 360, data_array)

# Function to parse file metadata (variable, model, concentration, period) from filename
def parse_filename(filename):
    pattern = re.compile(r"_([a-zA-Z0-9]+)_([a-zA-Z0-9\-]+)_r[0-9]+i[0-9]+p[0-9]+_(historical|rcp45|rcp85)_(\d{4})_(\d{4})")
    match = pattern.search(filename)
    return match.groups() if match else (None, None, None, None, None)

def split_and_filter_files(files, model, concentration, target_start, target_end):
    """
    Filter files based on model, concentration, and split them to match the target period.
    """
    matching_files = []
    for file in files:
        variable, file_model, file_concentration, file_start, file_end = parse_filename(file)
        file_start, file_end = int(file_start), int(file_end)

        if file_model == model and file_concentration == concentration:
            # Check if the file overlaps with the target period
            if file_start <= target_end and file_end >= target_start:
                matching_files.append((file, max(file_start, target_start), min(file_end, target_end)))
    return matching_files

# Function to calculate relative humidity (RHU)
def qair2rh(qair, temp, press=1013.25):
    """Convert specific humidity to relative humidity."""
    es = 6.112 * np.exp((17.67 * temp) / (temp + 243.5))  # Saturation vapor pressure
    e = qair * press / (0.378 * qair + 0.622)            # Actual vapor pressure
    rh = e / es                                          # Relative humidity
    return np.clip(rh, 0, 1)                             # Limit between 0 and 1


def process_files_simple(files_with_periods, model, concentration, start_year, end_year, point_lat, point_lon):
    """
    Process NetCDF files for a given model, concentration, and time period into a combined DataFrame.
    """
    combined_data = pd.DataFrame()

    for file in files_with_periods:
        file_path = os.path.join(data_dir, file)

        try:
            # Open the dataset
            dataset = xr.open_dataset(file_path)
            dataset["lon"] = adjust_longitudes(dataset["lon"])
            
            if len(set(dataset["time"].values)) == 1:
                print(f"⚠ Warning: Invalid time index detected in {file}. skipping...")
                continue 

            # Filter by time range
            dataset = dataset.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
            data_point = dataset.sel(lat=point_lat, lon=point_lon, method="nearest")
            
            time = pd.to_datetime(data_point["time"].values)
            year_doy = pd.Series(time).dt.year * 1000 + pd.Series(time).dt.dayofyear

            # Extract variable from the filename
            variable, _, _, _, _ = parse_filename(file)
            actual_variable = variable_mapping.get(variable, variable)

            if actual_variable in dataset:
                # Extract and convert the variable
                converted_values = convert_units(variable, data_point[actual_variable].values)
                file_data = pd.DataFrame({
                    "yyddd": year_doy,
                    variable: converted_values,
                    "lat": data_point["lat"].values,
                    "lon": data_point["lon"].values
                })

                # Merge the new variable into the combined DataFrame
                if combined_data.empty:
                    combined_data = file_data
                else:
                    combined_data = pd.merge(
                        combined_data,
                        file_data,
                        on=["yyddd", "lat", "lon"],
                        how="outer"
                    )

                    # Handle duplicate columns explicitly
                    duplicate_cols = [col for col in combined_data.columns if "_x" in col or "_y" in col]
                    for col in duplicate_cols:
                        base_col = col.split("_")[0]
                        if base_col in combined_data:
                            # Combine non-empty entries from duplicate columns
                            combined_data[base_col] = combined_data[base_col].fillna(combined_data[col])
                        else:
                            # If base column doesn't exist, rename the duplicate column
                            combined_data.rename(columns={col: base_col}, inplace=True)
                        # Drop the processed duplicate column safely
                        if col in combined_data.columns:
                            combined_data.drop(columns=[col], inplace=True)

        except Exception as e:
            print(f"Error processing file {file}: {e}")

    return combined_data  # Return the processed DataFrame instead of saving it directly
 
