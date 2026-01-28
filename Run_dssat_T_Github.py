# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 08:41:51 2024

@author: Ahmed.Attia
"""
import os
import pandas as pd
import numpy as np
import yaml
import datetime
import subprocess
import re
import netCDF4 as netcdf
import sys, os, glob, gc, csv
from pylab import *
import datetime
from datetime import date
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
pd.set_option("display.max_columns", None)


spatial_data_all = pd.read_csv("C:/Users/ahmed.attia/OneDrive - Texas A&M AgriLife/SpatialData_TX/CSV_files/POI18by18kmHigh&RollingPlains1.csv")
spatial_pointsTX = spatial_data_all[spatial_data_all["US_L3NAME"] == "High Plains"]
spatial_pointsTX = spatial_pointsTX.sample(n=10)
#spatial_data_all['sitenumber'] = range(1,len(spatial_data_all) +1)  
#spatial_data_all.insert(0,"ID",generate_four_letter_ids(len(spatial_data_all)))        
Rot_design = pd.read_csv("C:/Users/ahmed.attia/OneDrive - Texas A&M AgriLife/SpatialData_TX/CSV_files/Rot_design.csv")
co2_data = pd.read_csv("D:/KlimaFFolgen/RCP_wise_CO2_RCP45.csv")
wisedata = pd.read_csv("C:/Users/ahmed.attia/OneDrive - Texas A&M AgriLife/SpatialData_TX/CSV_files/wise_sol_data.csv")   #wise soil data GE
#wisedata['ID'] = wisedata['ID'].str.replace('["\"*]','', regex = True)      
wisedata = wisedata.rename(columns={'X':'Lon','Y':'Lat','SoilProfil':'ID'})                                             
with open("C:/DSSAT48/Soil/US.SOL", "r", encoding="utf-8") as f:
    US_sol = f.readlines()
US_sol_trimed = [line.rstrip() for line in US_sol]
del US_sol

#---lists----
sum_his = {}     # 1990 - 2005
sum_mid = {}     # 2006 - 2052
sum_end = {}     # 2053 - 2099
sum_hmf = {}     # 1990 - 2099

sum_his_site = {}     # 1990 - 2005
sum_mid_site = {}     # 2006 - 2052
sum_end_site = {}     # 2053 - 2099
sum_hmf_site = {}     # 1990 - 2099

sum_his_model = {}     # 1990 - 2005
sum_mid_model = {}     # 2006 - 2052
sum_end_model = {}     # 2053 - 2099
sum_hmf_model = {}     # 1990 - 2099
delta_hmf = {}     # 1990-2099
delta_hmf_site = {}
delta_hmf_model = {}

GHG_bal_yr = {}  # yearly_GHG_1991_2099
GHG_bal_yr_site = {}
GHG_bal_yr_model = {}

climate_models = [ "CCSM4","CNRM-CM5","CSIRO-Mk3-6-0"]
data_dir = "C://Gdrive//AOI_Climate_data/"  # Replace with your actual folder path
all_files = [f for f in os.listdir(data_dir) if f.endswith(".nc")]
#s, row = next(spatial_pointsTX.iterrows())

#-------------main for loop--------------
for model in climate_models:
    print(f"Processing model: {model}")
    
    for s, row in spatial_pointsTX.iterrows():  #  Loop over multiple spatial points and keep the row index 's' instead of '_'
    
        point_lat, point_lon = row["point_lat"],row["point_lon"]
        print(f"Processing location: lat={point_lat}, lon={point_lon}")
    
        # Process Historical Period
        historical_files = [f for f in all_files if "historical" in f and model in f]
        historical_data = process_files_simple(historical_files, model, "historical", 1990, 2005, point_lat, point_lon)
    
        # Process RCP45
        rcp45_files = [f for f in all_files if "rcp45" in f and model in f]
        rcp45_data = process_files_simple(rcp45_files, model, "rcp45", 2006, 2099, point_lat, point_lon)
    
        # Append Historical Data to RCP45 and RCP85
        if not historical_data.empty:
            if not rcp45_data.empty:
                rcp45_combined = pd.concat([historical_data, rcp45_data]).sort_values("yyddd")
                # Compute Relative Humidity (RHU) and add to combined_data
                try:
                    if "huss" in rcp45_combined and "tasmax" in rcp45_combined and "tasmin" in rcp45_combined:
                        temp_mean = (rcp45_combined["tasmax"] + rcp45_combined["tasmin"]) / 2  # Average temperature
                        rcp45_combined["RHU"] = qair2rh(rcp45_combined["huss"], temp_mean, press=P) * 100
                except Exception as e:
                    print(f"Error calculating RHU: {e}")
                    
    
                # Rename columns to match the desired order
                column_rename_mapping = {
                    'yyddd': 'yyddd',
                    'rsds': 'SRAD',
                    'tasmax': 'TMAX',
                    'tasmin': 'TMIN',
                    'pr': 'RAIN',
                    'was': 'WIND',
                    'RHU': 'RHUM'
                }
                rcp45_combined = rcp45_combined.rename(columns=column_rename_mapping)
    
                # Define the desired column order
                desired_order = ['lat', 'lon', 'yyddd', 'SRAD', 'TMAX', 'TMIN', 'RAIN', 'WIND', 'RHUM']
    
                # Reorder columns (ensure all columns in the order exist in the DataFrame)
                rcp45_combined = rcp45_combined[[col for col in desired_order if col in rcp45_combined.columns]]
                
                local_data =  rcp45_combined.copy()
                local_data = adjust_temperatures(local_data)

                local_data["DATE"] = pd.to_datetime(local_data["yyddd"].astype(str), format="%Y%j", errors='coerce')

                wthdir = "C:/DSSAT48/Weather"  # Output directory
                station_name = spatial_pointsTX['ID'][s]  # 4-character DSSAT station name
                
                write_dssat_wth(
                    local_data=local_data, 
                    xy=(local_data["lon"].iloc[0], local_data["lat"].iloc[0]), 
                    elev=309,  # Example elevation
                    co2_data=co2_data, 
                    outdir=wthdir, 
                    name=station_name
                )
                
        wisedata['distance'] = ((spatial_pointsTX['point_lat'][s] - wisedata['Lat'])**2 + (spatial_pointsTX['point_lon'][s] - wisedata['Lon'])**2)**0.5 * 110
        sorted_wisedata = wisedata.sort_values(by='distance')
        nearestgrid = sorted_wisedata['ID'].iloc[0]
        prof_loc = [i for i, line in enumerate(US_sol_trimed) if re.match(rf"^\*{nearestgrid}", line)]
    
        for k in range(len(Rot_design)):
            
            try:
                
                if Rot_design['Expname'][k] in Rot_design['Expname'][0]:
                           
                    # Assuming `Rot_design` is a pandas DataFrame and `k` is an integer 
                    TREF1111_MZX = Rot_design['Expname'][k]
                    Trt = list(range(1, 15))
                    
                    out_stringb = ("$BATCH(SEQUENCE)\n"
                                   "!\n"
                                   "! Directory    : C:\\DSSAT48\\Sequence\n"
                                   "! Command Line : C:\\DSSAT48\\DSCSM048.EXE Q DSSBatch.v48\n"
                                   "! Crop         : Sequence\n"
                                   f"! Experiment   : {Rot_design['Expname'][k]}\n"
                                   "! ExpNo        : 1\n"
                                   "! Debug        : C:\\DSSAT48\\DSCSM048.EXE ' Q DSSBatch.v48\n"
                                   "!\n"
                                   "@FILEX                                                                                       TRTNO     RP     SQ     OP     CO\n")
                    
                    TrtR = []
                    for a in Trt:
                        Trtp = f"C:\\DSSAT48\\Sequence\\{Rot_design['Expname'][k]}                                                                1       1     {a:02d}      0      0\n"
                        TrtR.append(Trtp)
                    
                    TrtR2 = [line.replace("TREF1111.MZX", TREF1111_MZX) for line in TrtR]
                    
                    with open("C:/DSSAT48/Sequence/DSSBatch.v48", "w") as f:
                        f.write(out_stringb)
                        f.write("".join(TrtR2))
                        
                    expname = Rot_design['Expname'][k]   
                    # Extract specific lines based on the profile location
                    line_index = prof_loc[0]
                    site_5 = US_sol_trimed[line_index + 6]
                    site_15 = US_sol_trimed[line_index + 7]
                    site_30 = US_sol_trimed[line_index + 8]
                    site_60 = US_sol_trimed[line_index + 9]
                    site_100 = US_sol_trimed[line_index + 10]
                    site_200 = US_sol_trimed[line_index + 11]
                    
                    # Function to extract numeric values from a string
                    def extract_numbers(line):
                        return np.array([float(num) for num in re.findall(r"\d+\.\d+", line)])
                    
                    # Extract silt and clay values
                    silt_clay5 = extract_numbers(site_5)[6:9]
                    silt_clay15 = extract_numbers(site_15)[6:9]
                    silt_clay30 = extract_numbers(site_30)[6:9]
                    silt_clay60 = extract_numbers(site_60)[6:9]
                    silt_clay100 = extract_numbers(site_100)[6:9]
                    silt_clay200 = extract_numbers(site_200)[6:9]
                    
                    # Function to compute stable fraction
                    def compute_stable_fraction(silt_clay):
                        stable = (0.015 * (silt_clay[1] + silt_clay[2]) + 0.069) / silt_clay[0]
                        return np.clip(stable, 0.1, 0.7)  # Equivalent to pmin(pmax(..., 0.1), 0.8) in R
                    
                    # Compute stable fractions
                    stable5 = compute_stable_fraction(silt_clay5)
                    stable15 = compute_stable_fraction(silt_clay15)
                    stable30 = compute_stable_fraction(silt_clay30)
                    stable60 = compute_stable_fraction(silt_clay60)
                    stable100 = compute_stable_fraction(silt_clay100)
                    stable200 = compute_stable_fraction(silt_clay200)
                     
                    # Read the DSSAT sequence file
                    sequence_file = f"C:/DSSAT48/Sequence/{Rot_design['Expname'][k]}"
                    with open(sequence_file, "r") as f:
                        xbuild = f.readlines()
                    
                    # Find line numbers for "*FIELDS" and "*SIMULATION CONTROLS"
                    line_number = next(i for i, line in enumerate(xbuild) if "*FIELDS" in line)
                    sim_length = next(i for i, line in enumerate(xbuild) if "*SIMULATION CONTROLS" in line)
                    
                    # Update the FIELDS section
                    xbuild[line_number + 2] = f" 1 BS01b4FF {spatial_pointsTX['ID'][s]}9099   -99   -99   -99   -99   -99     0 LS     190  {nearestgrid} Rotation number {Rot_design['Rot_number'][k]}\n"
                    
                    # Update the SIMULATION CONTROLS section
                    xbuild[sim_length + 4] = " 1 OP              Y     Y     Y     N     N     N     N     Y     W\n"
                    
                    # Find the soil profile section
                    solAna = next(i for i, line in enumerate(xbuild) if re.match(r"^\*SOIL", line))
                    
                    # Update soil data with formatted stable values
                    xbuild[solAna + 4] = f" 1     5     1  2.01   .04   7.6   -99    29   222  {stable5:4.2f}\n"
                    xbuild[solAna + 5] = f" 1    15     1  1.51   .03   7.7   -99    29   222  {stable15:4.2f}\n"
                    xbuild[solAna + 6] = f" 1    30     1  1.01   .04   7.7   -99    17   178  {stable30:4.2f}\n"
                    xbuild[solAna + 7] = f" 1    60     1   .55   .04   7.8   -99     5   168  {stable60:4.2f}\n"
                    xbuild[solAna + 8] = f" 1   100    .4   .35   .04   7.9   -99     4   172  {stable100:4.2f}\n"
                    xbuild[solAna + 9] = f" 1   200    .2   .24   .03   8.2   -99     5   178  {stable200:4.2f}\n"
                    
                    # Save the modified file
                    output_file = f"C:/DSSAT48/Sequence/{Rot_design['Expname'][k]}"
                    with open(output_file, "w") as f:
                        f.writelines(xbuild)
    
                    os.chdir("C:/DSSAT48/Sequence")
                    command = "C:/DSSAT48/DSCSM048.EXE Q DSSBatch.v48"
                    subprocess.run(command, shell=True, capture_output=False)
                    
                    summary_file = "C:/DSSAT48/Sequence/Summary.OUT"
                    
                    summ_raw = read_dssat_summary(summary_file)
                    # Rename columns
                    summ_raw.rename(columns={
                        summ_raw.columns[7]: "EXNAME",
                        summ_raw.columns[8]: "TNAM",
                        summ_raw.columns[9]: "FNAM",
                        summ_raw.columns[10]: "WSTA",
                        summ_raw.columns[12]: "SOIL_ID"
                    }, inplace=True)
                    # Trim whitespace in TNAM
                    summ_raw["TNAM"] = summ_raw["TNAM"].str.strip()
                    # Apply the date conversion for relevant columns
                    for col in ["HDAT", "SDAT", "PDAT"]:
                        summ_raw[col] = summ_raw[col].apply(convert_dssat_date)
                        
                    summ_i = summ_raw.copy()
                    summ_i["climate"] = model
                    summ_i["site"] = spatial_pointsTX["Sitenumber"][s]
                    summ_i["lon"] = spatial_pointsTX["point_lon"][s]
                    summ_i["lat"] = spatial_pointsTX["point_lat"][s]
                    summ_i["soil_id"] = nearestgrid
                    summ_i["Rotation.number"] = Rot_design["Rot_number"][k]
                    summ_i["Rotation.type"] = Rot_design["rot_type"][k]
                    summ_i["code"] = Rot_design["code"][k]
                    summ_i["Num_CCs"] = Rot_design["num_CCs"][k]
                
                    summ1990_2099 = summ_i.copy()
                    summ1990_2005 = summ_i[(summ_i["HDAT"] >= "1990-01-01") & (summ_i["HDAT"] <= "2005-12-31")]
                    summ2006_2052 = summ_i[(summ_i["HDAT"] >= "2006-01-01") & (summ_i["HDAT"] <= "2052-12-31")]
                    summ2053_2099 = summ_i[(summ_i["HDAT"] >= "2053-01-01") & (summ_i["HDAT"] <= "2099-12-31")]
                    
                    summ_his = compute_summary(summ1990_2005)
                    summ_mid = compute_summary(summ2006_2052)
                    summ_end = compute_summary(summ2053_2099)
                    summ_hmf = compute_summary(summ1990_2099)
                    group_cols = ["climate", "site", "soil_id", "Rotation.number", "code", "lon", "lat"]
                    grouping_info = summ1990_2005[group_cols].drop_duplicates().reset_index(drop=True)
    
                    sum_his[k] = pd.concat([grouping_info,summ_his],axis=1)
                    sum_mid[k] = pd.concat([grouping_info,summ_mid],axis=1)
                    sum_end[k] = pd.concat([grouping_info,summ_end],axis=1)
                    sum_hmf[k] = pd.concat([grouping_info,summ_hmf],axis=1)
                    
                    OrgN = 'C:/DSSAT48/Sequence/SoilOrg.OUT'
                    fieldVec = ["YEAR","DOY","ONAC"]
                    results = read_dssat_df(fileVec=[OrgN],fieldVec=fieldVec)
                    # Extract dictionary of DataFrames from the list
                    results_dict = results[0]  # Extract the single dictionary from the list
    
                    # Convert dictionary values to a list of DataFrames
                    dfs = list(results_dict.values())
    
                    # Define required columns
                    required_columns = ["YEAR","DOY","ONAC"]
    
                    # Filter DataFrames to keep only available columns
                    dfs = [df[[col for col in required_columns if col in df.columns]] for df in dfs]
    
                    # Merge all DataFrames while avoiding duplicate columns
                    OrgN_df1 = next((df for df in dfs if not df.empty), None)
                    OrgN_df1 = OrgN_df1.iloc[1:,: ]
                    OrgN_df1 = OrgN_df1.groupby('YEAR',as_index=False).agg({"ONAC":"max"})
    
                    Napp = 'C:/DSSAT48/Sequence/SoilNi.OUT'
                    fieldVec = ["YEAR","DOY","NAPC","NLCC"]
                    results = read_dssat_df(fileVec=[Napp],fieldVec=fieldVec)
                    # Extract dictionary of DataFrames from the list
                    results_dict = results[0]  # Extract the single dictionary from the list
     
                    # Convert dictionary values to a list of DataFrames
                    dfs = list(results_dict.values())
     
                    # Define required columns
                    required_columns = ["YEAR","DOY","NAPC","NLCC"]
     
                    # Filter DataFrames to keep only available columns
                    dfs = [df[[col for col in required_columns if col in df.columns]] for df in dfs]
     
                    # Merge all DataFrames while avoiding duplicate columns
                    # Find the first non-empty DataFrame
                    Napp_df2 = next((df for df in dfs if not df.empty), None)
                    Napp_df2 =Napp_df2.iloc[1:,:]
                    Napp_df2 = Napp_df2.groupby('YEAR',as_index=False).agg({"NAPC":"max","NLCC":"max"})
                    
                    plantN = 'C:/DSSAT48/Sequence/PlantN.OUT'
                    fieldVec = ["YEAR","DOY","CNAD","GNAD","HNAD","RN%D"]
                    results = read_dssat_df(fileVec=[plantN],fieldVec=fieldVec)
                    # Extract dictionary of DataFrames from the list
                    results_dict = results[0]  # Extract the single dictionary from the list
     
                    # Convert dictionary values to a list of DataFrames
                    dfs = list(results_dict.values())
     
                    # Define required columns
                    required_columns = ["YEAR","DOY","CNAD","GNAD","HNAD","RN%D"]
     
                    # Filter DataFrames to keep only available columns
                    dfs = [df[[col for col in required_columns if col in df.columns]] for df in dfs]
     
                    # Merge all DataFrames while avoiding duplicate columns
                    # Find the first non-empty DataFrame
                    plantN_df3 = next((df for df in dfs if not df.empty), None)
                    
                    for i, df in enumerate(dfs[1:], start=2):
                        plantN_df3 = pd.merge(
                            plantN_df3, df, on=["YEAR", "DOY"], how="outer",
                            suffixes=(None, f"_{i}")  # Rename duplicate columns
                        )
    
                    # Combine duplicate columns (merge GNAD_x, GNAD_y into one)
                    for col in required_columns:
                        duplicate_cols = [c for c in plantN_df3.columns if c.startswith(col + "_")]
                        if duplicate_cols:
                            plantN_df3[col] = plantN_df3[col].combine_first(plantN_df3[duplicate_cols].bfill(axis=1).iloc[:, 0])
                            plantN_df3.drop(columns=duplicate_cols, inplace=True)
    
                    # Sort values by YEAR and DAS
                    plantN_df3 = plantN_df3.sort_values(by=["YEAR", "DOY"]).reset_index(drop=True)
                    plantN_df3['GNHNAD'] = plantN_df3['GNAD'].fillna(plantN_df3['HNAD'])
                    
                    plantGro = 'C:/DSSAT48/Sequence/PlantGro.OUT'
                    fieldVec = ["YEAR","DOY","RWAD"]
                    results = read_dssat_df(fileVec=[plantGro],fieldVec=fieldVec)
                    # Extract dictionary of DataFrames from the list
                    results_dict = results[0]  # Extract the single dictionary from the list
     
                    # Convert dictionary values to a list of DataFrames
                    dfs = list(results_dict.values())
     
                    # Define required columns
                    required_columns = ["YEAR","DOY","RWAD"]
     
                    # Filter DataFrames to keep only available columns
                    dfs = [df[[col for col in required_columns if col in df.columns]] for df in dfs]
     
                    # Merge all DataFrames while avoiding duplicate columns
                    # Find the first non-empty DataFrame
                    plantGro_df4 = next((df for df in dfs if not df.empty), None)
                    
                    for i, df in enumerate(dfs[1:], start=2):
                        plantGro_df4 = pd.merge(
                            plantGro_df4, df, on=["YEAR", "DOY"], how="outer",
                            suffixes=(None, f"_{i}")  # Rename duplicate columns
                        )
    
                    # Combine duplicate columns (merge GNAD_x, GNAD_y into one)
                    for col in required_columns:
                        duplicate_cols = [c for c in plantGro_df4.columns if c.startswith(col + "_")]
                        if duplicate_cols:
                            plantGro_df4[col] = plantGro_df4[col].combine_first(plantGro_df4[duplicate_cols].bfill(axis=1).iloc[:, 0])
                            plantGro_df4.drop(columns=duplicate_cols, inplace=True)
    
                    # Sort values by YEAR and DAS
                    plantGro_df4 = plantGro_df4.sort_values(by=["YEAR", "DOY"]).reset_index(drop=True)
                    
                    plantN_df5 = pd.DataFrame({"Yr": plantGro_df4['YEAR'], "GrianN" : plantN_df3['GNHNAD'],"topsN": plantN_df3['CNAD'],
                                               "RootsN" : (plantGro_df4['RWAD']*plantN_df3['RN%D'])/100})
                    plantN_df5 = plantN_df5.groupby('Yr',as_index=False).agg({"GrianN":"max","topsN":"max","RootsN":"max"})
                    
                    soilSOC = 'C:/DSSAT48/Sequence/SOMLITC.OUT'
                    fieldVec = ["YEAR","DOY","SOCD"]
                    results = read_dssat_df(fileVec=[soilSOC],fieldVec=fieldVec)
                    # Extract dictionary of DataFrames from the list
                    results_dict = results[0]  # Extract the single dictionary from the list
    
                    # Convert dictionary values to a list of DataFrames
                    dfs = list(results_dict.values())
    
                    # Define required columns
                    required_columns = ["YEAR","DOY","SOCD"]
    
                    # Filter DataFrames to keep only available columns
                    dfs = [df[[col for col in required_columns if col in df.columns]] for df in dfs]
    
                    # Merge all DataFrames while avoiding duplicate columns
                    soilSOC_df1 = next((df for df in dfs if not df.empty), None)
                    soilSOC_df1 = soilSOC_df1.groupby('YEAR',as_index=False).agg({"SOCD":"mean"})
                    
                    aboveN = plantN_df5['topsN']-plantN_df5['GrianN']
                    belowN = plantN_df5['RootsN']
                    N_app = Napp_df2['NAPC']
                    Org_N = OrgN_df1['ONAC']
                    N_leach = Napp_df2['NLCC']
                    
                    N2O_CO2eq = N2OEmission(aboveN, belowN,
                                     N_app, Org_N,N_leach, tier = 1, EF_regional=None, co2e = True)
                    N2O_CO2eq = N2O_CO2eq[1:]
                    
                    SOC_CO2eq_cha_prof = soilSOC_df1["SOCD"].diff()[2:]*44/12
                    GHG_cha_prof = N2O_CO2eq - SOC_CO2eq_cha_prof
    
                    GHG_balancedata = pd.DataFrame({
                            "yr": soilSOC_df1["YEAR"].iloc[2:],  # Drop the first row like `[-1]`
                            "SOC_CO2eq_prof": np.round(SOC_CO2eq_cha_prof, 3),
                            "tSOC_CO2eq_prof/ha": np.round(np.cumsum(SOC_CO2eq_cha_prof) / 1000, 3),
                            "N2O_CO2eq": np.round(N2O_CO2eq, 3),
                            "tN2O_CO2eq/ha": np.round(np.cumsum(N2O_CO2eq) / 1000, 3),
                            "GHG_prof(kgco2eq/ha/yr)": np.round(GHG_cha_prof, 3),
                            "cumGHG_prof(tco2eq/ha)":np.round(np.cumsum(GHG_cha_prof) / 1000, 3),
                            "kgGHG/GEunit":  np.round(GHG_cha_prof / sum_hmf[k]["Rot_GE_m"].iloc[0], 3)  # More precise rounding,
                        })

                    GHG_bal_yr[k] = pd.concat([
                        pd.DataFrame({
                            "climate": [model] * len(GHG_balancedata),
                            "site": [spatial_pointsTX["Sitenumber"][s]] * len(GHG_balancedata),
                            "lon": [spatial_pointsTX["point_lon"][s]] * len(GHG_balancedata),
                            "lat": [spatial_pointsTX["point_lat"][s]] * len(GHG_balancedata),
                            "Rotation.number": [Rot_design["Rot_number"][k]] * len(GHG_balancedata),
                            "code": [Rot_design["code"][k]] * len(GHG_balancedata),
                            "Num_CCs": [Rot_design["num_CCs"][k]] * len(GHG_balancedata),    
                            }),
                        GHG_balancedata.reset_index(drop=True)  # Ensure indices match
                        ], axis=1)
                    
                    # Compute values
                    delta_hmf[k] = pd.DataFrame({
                        "climate": [model],
                        "site": [spatial_pointsTX["Sitenumber"][s]],
                        "lon": [spatial_pointsTX["point_lon"][s]],
                        "lat": [spatial_pointsTX["point_lat"][s]],
                        "code": [Rot_design["code"][k]],
                        "Rotation.number": [Rot_design["Rot_number"][k]],
                        
                        # Compute means and sums
                        "SOC_CO2eqProf": [round(SOC_CO2eq_cha_prof.mean(), 3)],
                        "N2O_CO2eq": [round(N2O_CO2eq.mean(), 3)],
                        "GHG_prof_kgCO2eq/ha/yr": [round(GHG_cha_prof.mean(), 3)],
                        "GHG_prof(tco2eq/ha)":[round(GHG_cha_prof.sum() / 1000, 3)],
                        
                        # Compute kgGHG per unit energy
                        "kgGHG/GEunit": [round(GHG_cha_prof.sum() / sum_hmf[k]["Rot_GEs"].iloc[0], 3)],  
                    })  # Extract scalar value
                                          
              ####postprocessing calculations continue
            ######hold block####
                     
               
            except Exception as e:
                print(f"Error processing experiment {k}: {e}")
                
            sum_his_site[s] = pd.concat(sum_his.values(),ignore_index=True)
            sum_mid_site[s] = pd.concat(sum_mid.values(),ignore_index=True)  
            sum_end_site[s] = pd.concat(sum_end.values(),ignore_index=True)
            sum_hmf_site[s] = pd.concat(sum_hmf.values(),ignore_index=True)
            
            delta_hmf_site[s] = pd.concat(delta_hmf.values(),ignore_index=True)
            GHG_bal_yr_site[s] = pd.concat(GHG_bal_yr.values(), ignore_index=True)
            
        sum_his_model[model] = pd.concat(sum_his_site.values(),ignore_index=True)
        sum_mid_model[model] = pd.concat(sum_mid_site.values(),ignore_index=True)  
        sum_end_model[model] = pd.concat(sum_end_site.values(),ignore_index=True)
        sum_hmf_model[model] = pd.concat(sum_hmf_site.values(),ignore_index=True)
            
        delta_hmf_model[model] = pd.concat(delta_hmf_site.values(), ignore_index=True)
        GHG_bal_yr_model[model] = pd.concat(GHG_bal_yr_site.values(), ignore_index=True)
        
    

sum_his_model_combined = pd.concat(sum_his_model.values(), ignore_index=True)
sum_mid_model_combined = pd.concat(sum_mid_model.values(), ignore_index=True)
sum_end_model_combined = pd.concat(sum_end_model.values(), ignore_index=True)
sum_hmf_model_combined = pd.concat(sum_hmf_model.values(), ignore_index=True)
      
delta_hmf_combined = pd.concat(delta_hmf_model.values(), ignore_index=True)
GHG_bal_yr_model_combined = pd.concat(GHG_bal_yr_model.values(), ignore_index=True)

output_folder = "C:/Users/ahmed.attia/OneDrive - Texas A&M AgriLife/Results/Simulation_outputs" 
sum_his_model_combined.to_csv(os.path.join(output_folder, "sum_his_modelmodelrcp45.csv"), index=False) 
sum_mid_model_combined.to_csv(os.path.join(output_folder, "sum_mid_modelrcp45.csv"), index=False)
sum_end_model_combined.to_csv(os.path.join(output_folder, "sum_end_modelrcp45.csv"), index=False) 
sum_hmf_model_combined.to_csv(os.path.join(output_folder, "sum_hmf_modelrcp45.csv"), index=False)           
delta_hmf_combined.to_csv(os.path.join(output_folder, "delta_hmfrcp45.csv"), index=False) 
GHG_bal_yr_model_combined.to_csv(os.path.join(output_folder, "GHG_bal_yr_modelrcp45.csv"), index=False)          
           
                
               
                
    
                
                 
                 
                
                
                
                
                
                
                                    
                

               
                
                
                



                    


                
                
                
                
                    
                    
                    
                   