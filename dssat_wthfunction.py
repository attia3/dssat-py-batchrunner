import pandas as pd
import numpy as np
import os



def adjust_temperatures(local_data):
    # Check if TMAX is always greater than TMIN
    if (local_data['TMAX'] > local_data['TMIN']).all():
        print("TMAX is always greater than TMIN.")
    else:
        print("TMAX is not always greater than TMIN. Adjusting temperatures...")
        
        # Adjust TMAX and TMIN as necessary
        local_data['TMAX'] = np.maximum(local_data['TMAX'], local_data['TMIN'])
        local_data['TMIN'] = np.where(local_data['TMAX'] <= local_data['TMIN'], local_data['TMIN'] - 1.2, local_data['TMIN'])
    
    return local_data


def write_dssat_wth(local_data, xy, elev, co2_data, outdir, name):
    """
    Write a DSSAT weather (.WTH) file from processed climate data.
    
    Parameters:
    - local_data: DataFrame containing weather data.
    - xy: Tuple (longitude, latitude) of the weather station.
    - elev: Elevation (m).
    - co2_data: DataFrame with 'Year' and 'CO2' columns.
    - outdir: Directory to save the output file.
    - name: Weather station name (4 characters).
    """

    # Ensure correct output directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Extract year and DOY from yyddd
    local_data["YEAR"] = local_data["yyddd"].astype(str).str[:4].astype(int)
    local_data["DOY"] = local_data["yyddd"].astype(str).str[4:].astype(int)

    # Match CO2 values to each year
    local_data["CO2"] = local_data["YEAR"].map(lambda yr: co2_data.loc[co2_data["Year"] == yr, "CO2"].values[0] if yr in co2_data["Year"].values else -99)

    # Ensure required columns exist, fill missing with default DSSAT values (-99)
    required_cols = ["SRAD", "TMAX", "TMIN", "RAIN", "WIND", "RHUM", "CO2"]
    for col in required_cols:
        if col not in local_data.columns:
            local_data[col] = -99  # Default missing values

    # Correct DSSAT filename format (e.g., AAAE9099.WTH)
    if len(name) == 4:
        first_year = str(local_data['YEAR'].iloc[0])[2:]  # Extract last two digits of first year
        last_year = str(local_data['YEAR'].iloc[-1])[2:]  # Extract last two digits of last year
        outname = f"{name}{first_year}{last_year}.WTH"
    elif len(name) == 8:
        outname = f"{name}.WTH"
    else:
        raise ValueError("Weather station name must be 4 or 8 characters long.")

    # Compute monthly temperature mean
    local_data["MONTH"] = pd.to_datetime(local_data["DATE"]).dt.month
    mo_mu = local_data.groupby("MONTH")[["TMAX", "TMIN"]].mean().mean(axis=1)

    # Header
    header = [
    f"$WEATHER DATA : {name}",
    "",
    "@ INSI      LAT     LONG  ELEV   TAV   AMP REFHT WNDHT",
    f"  {name:<6}{xy[1]:7.3f}{xy[0]:9.3f}{elev:6.0f}{mo_mu.mean():6.1f}{(mo_mu.max() - mo_mu.min()) * 0.5:6.1f}  1.75  2.00"
    ]

    # Weather data
    weather_data = local_data[["yyddd", "SRAD", "TMAX", "TMIN", "RAIN", "WIND", "RHUM", "CO2"]]
    weather_data_str = weather_data.to_string(index=False, header=["@  DATE", "SRAD", "TMAX", "TMIN", "RAIN", "WIND", "RHUM", "CO2"], float_format="{:.1f}".format)

    # Combine all text
    file_content = "\n".join(header) + "\n" + weather_data_str

    # Save to file
    file_path = os.path.join(outdir, outname)
    with open(file_path, "w") as f:
        f.write(file_content)

    print(f"Writing DSSAT weather file for: {name}")
