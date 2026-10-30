# -*- coding: utf-8 -*-
"""
=========================================================================
Script Name: Interannual_STD_Calculation_Junggar.py
Description: Blockwise calculation of inter-annual standard deviation for 
             vegetation/moisture indices (NDVI/NDMI) using
             multi-year multi-band GeoTIFF time series data.
Study Area:  Arid Region of Northwest China (ARNC) - Junggar Basin (Basin II)
Author:      Jie Zhou (zhenleiyang@ms.xjb.ac.cn)
Reference:   Zhou et al., Scientific Data (Nature Portfolio), 2026.
Environment: Python 3.10+ (rasterio, numpy)
=========================================================================
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window

# ================= Configuration & Parameter Settings =================
# # 1. Input: Multi-band GeoTIFF paths for years 2000-2004, ordered chronologically
# file_list = [
#     r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\MaskVg2000NDMI.tif",
#     r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\MaskVg2001NDMI.tif",
#     r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\MaskVg2002NDMI.tif",
#     r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\_2003NDMI0811vgGEMunion.tif",
#     r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\_2004_NDMI_SG_0809new12vgGEMunion.tif",
# ]

file_list = [
    r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\MaskVg2000NDVI.tif",
    r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\MaskVg2001NDVI.tif",
    r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\MaskVg2002NDVI.tif",
    r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\_2003NDVI0811vgGEMunion.tif",
    r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\_2004_NDVI_SG_0809new12vgGEMunion.tif",
]

# file_list = [
#     r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\MaskVg2005NDMI.tif",
#     r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\MaskVg2006NDMI.tif",
#     r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\MaskVg2007NDMI.tif",
#     r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\MaskVg2008NDMI.tif",
#     r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\MaskVg2009NDMI.tif",
# ]

# file_list = [
#     r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\MaskVg2005NDVI.tif",
#     r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\MaskVg2006NDVI.tif",
#     r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\MaskVg2007NDVI.tif",
#     r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\MaskVg2008NDVI.tif",
#     r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\MaskVg2009NDVI.tif",
# ]

# file_list = [
#     r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\MaskVg2010NDMI.tif",
#     r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\MaskVg2011NDMI.tif",
#     r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\MaskVg2012NDMI.tif",
#     r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\MaskVg2013NDMI.tif",
#     r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\MaskVg2014NDMI.tif",
# ]

# file_list = [
#     r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\MaskVg2010NDVI.tif",
#     r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\MaskVg2011NDVI.tif",
#     r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\MaskVg2012NDVI.tif",
#     r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\MaskVg2013NDVI.tif",
#     r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\MaskVg2014NDVI.tif",
# ]

# 2. Target band range: Months 3 to 10 (1-based index, consistent with MATLAB)
month_bands = list(range(3, 11))  

# 3. Output directory and filename
out_dir = r"I:\GDEs_Data\SG_result_lt15\Basins_Output\zhungaer1DissInts\Extracted_1_2_5\GEM_MaskVg\STD_python"

out_file = os.path.join(out_dir, "MaskVg_2000-2004_Months_3-10_Interannual_STD_NDVI.tif")

# 4. Block size and NoData parameters (consistent with MATLAB setup)
block_size = 4000
nodata_val = -32768
# =================================================================

# ================= Validation and Metadata Reading =================
# Check whether all input files exist
for f in file_list:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Input file does not exist: {f}")

# Create output directory if it does not exist
os.makedirs(out_dir, exist_ok=True)

# Read spatial reference and metadata from the reference image
with rasterio.open(file_list[0]) as src:
    rows = src.height
    cols = src.width
    crs = src.crs
    transform = src.transform
    src_profile = src.profile.copy()
    num_bands = src.count

# Validate whether specified bands are within bounds
if max(month_bands) > num_bands:
    raise ValueError(f"Specified band indices exceed total band count (Total bands: {num_bands})")

print(f"Total input files: {len(file_list)}")
print(f"Raster dimensions: {rows} rows x {cols} cols, {num_bands} bands per file")
print(f"Calculation window: Bands {min(month_bands)}-{max(month_bands)} (Months {min(month_bands)}-{max(month_bands)})")
print(f"NoData value: {nodata_val}")
print("Starting inter-annual standard deviation calculation for multi-year growing seasons...")

# ================= Pre-allocate Global Result Matrix (float32 / single) =================
std_result = np.full((rows, cols), np.nan, dtype=np.float32)

# ================= Blockwise Processing =================
for row_start in range(0, rows, block_size):
    row_end = min(row_start + block_size, rows)
    block_h = row_end - row_start

    for col_start in range(0, cols, block_size):
        col_end = min(col_start + block_size, cols)
        block_w = col_end - col_start

        # Define processing window (col_off, row_off, width, height)
        window = Window(col_off=col_start, row_off=row_start,
                        width=block_w, height=block_h)

        # Pre-allocate array for annual growing-season means: (H, W, num_years)
        annual_mean = np.full((block_h, block_w, len(file_list)),
                              np.nan, dtype=np.float32)

        # Read bands and calculate growing season mean for each year
        for y_idx, file_path in enumerate(file_list):
            with rasterio.open(file_path) as src:
                # Read specified bands for current window: shape (bands, height, width)
                data = src.read(indexes=month_bands, window=window)
                data = data.astype(np.float32)

            # Mask NoData values
            data[data == nodata_val] = np.nan

            # Compute mean along band dimension (growing season mean): shape (height, width)
            annual_mean[:, :, y_idx] = np.nanmean(data, axis=0)

        # Calculate inter-annual population standard deviation (ddof=0 matches MATLAB std(..., 1))
        block_std = np.nanstd(annual_mean, axis=2, ddof=0)

        # Fill into global result matrix
        std_result[row_start:row_end, col_start:col_end] = block_std

    print(f"Processed rows: {row_end} / {rows}")

# ================= Result Summary Statistics =================
valid_pixels = np.sum(~np.isnan(std_result))
total_pixels = rows * cols
print(f"Valid pixels: {valid_pixels} / {total_pixels} ({100*valid_pixels/total_pixels:.2f}%)")

# ================= Export GeoTIFF (LZW Compression) =================
out_profile = src_profile
out_profile.update(
    count=1,
    dtype="float32",
    compress="lzw",  # LZW compression (corresponds to MATLAB Compression=5)
    nodata=np.nan    # Use NaN as NoData value for floating-point raster
)

with rasterio.open(out_file, "w", **out_profile) as dst:
    dst.write(std_result, 1)  # Write to band 1

print(f"Calculation completed successfully! Inter-annual STD result saved to: {out_file}")