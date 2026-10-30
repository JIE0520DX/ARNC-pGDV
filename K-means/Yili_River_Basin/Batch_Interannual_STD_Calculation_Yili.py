# -*- coding: utf-8 -*-
"""
=========================================================================
Script Name: Batch_Interannual_STD_Calculation_Yili.py
Description: Batch processing script for calculating the inter-annual standard
             deviation of vegetation/moisture indices (NDVI/NDMI)
             from multi-year multi-band GeoTIFF time series using blockwise reading.
Study Area:  Arid Region of Northwest China (ARNC) - Yili River Basin (Basin V)
Author:      Jie Zhou (zhenleiyang@ms.xjb.ac.cn)
Reference:   Zhou et al., Scientific Data (Nature Portfolio), 2026.
Environment: Python 3.10+ (rasterio, numpy)
=========================================================================
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window

# ================= 1. Core Calculation Function =================
def calculate_interannual_std(file_list, month_bands, out_file, block_size=4000, nodata_val=-32768):
    """
    Calculate the inter-annual population standard deviation of mean values
    for specified bands across a multi-year time series.
    """
    # Verify that all input files exist
    for f in file_list:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Input file does not exist: {f}")

    # Create output directory
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # Read reference file metadata
    with rasterio.open(file_list[0]) as src:
        rows = src.height
        cols = src.width
        src_profile = src.profile.copy()
        num_bands = src.count

    # Validate band index range
    if max(month_bands) > num_bands:
        raise ValueError(f"Specified month bands exceed total band count (current total bands: {num_bands})")

    print(f"--> Input files: {len(file_list)} | Dimensions: {rows}×{cols} | Processing bands: {month_bands}")

    # Pre-allocate output matrix
    std_result = np.full((rows, cols), np.nan, dtype=np.float32)

    # Blockwise processing
    for row_start in range(0, rows, block_size):
        row_end = min(row_start + block_size, rows)
        block_h = row_end - row_start

        for col_start in range(0, cols, block_size):
            col_end = min(col_start + block_size, cols)
            block_w = col_end - col_start

            window = Window(col_off=col_start, row_off=row_start, width=block_w, height=block_h)

            # Initialize annual growing-season mean array for current block
            annual_mean = np.full((block_h, block_w, len(file_list)), np.nan, dtype=np.float32)

            # Read each year and calculate growing-season mean
            for y_idx, file_path in enumerate(file_list):
                with rasterio.open(file_path) as src:
                    data = src.read(indexes=month_bands, window=window).astype(np.float32)
                
                # Mask NoData values
                data[data == nodata_val] = np.nan
                # Calculate mean along month bands -> (height, width)
                annual_mean[:, :, y_idx] = np.nanmean(data, axis=0)

            # Calculate inter-annual population standard deviation (ddof=0 corresponds to MATLAB population STD)
            std_result[row_start:row_end, col_start:col_end] = np.nanstd(annual_mean, axis=2, ddof=0)

        print(f"    Completed rows: {row_end} / {rows}", end="\r")
    print()  # Line break

    # Summary of valid pixels
    valid_pixels = np.sum(~np.isnan(std_result))
    total_pixels = rows * cols
    print(f"--> Valid pixel ratio: {valid_pixels}/{total_pixels} ({100*valid_pixels/total_pixels:.2f}%)")

    # Export to GeoTIFF (LZW compression)
    out_profile = src_profile
    out_profile.update(
        count=1,
        dtype="float32",
        compress="lzw",
        nodata=np.nan
    )

    with rasterio.open(out_file, "w", **out_profile) as dst:
        dst.write(std_result, 1)

    print(f"✅ Current task completed. Saved to: {out_file}\n")


# ================= 2. Batch Task Configuration =================
if __name__ == "__main__":
    # General parameter configuration
    month_bands = list(range(3, 12))  # Corresponding to bands 3-11 (e.g., March to November)
    block_size = 4000
    nodata_val = -32768
    out_dir = r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\STD_python"

    # Configure all batch tasks to process here
    batch_tasks = [
        # # --- Task 1: 2000-2004 NDMI ---
        # {
        #     "name": "2000-2004_NDMI",
        #     "file_list": [
        #         r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\MaskVg2000NDMI.tif",
        #         r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\MaskVg2001NDMI.tif",
        #         r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\MaskVg2002NDMI.tif",
        #         r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\_2003NDMI0811vgGEMUnion.tif",
        #         r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\_2004_NDMI_SG_0809new12vgGEMUnion.tif",
        #     ],
        #     "out_filename": "MaskVg_2000-2004_Months_3-11_Interannual_STD_NDMI.tif"
        # },
        # --- Task 2: 2000-2004 NDVI ---
        {
            "name": "2000-2004_NDVI",
            "file_list": [
                r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\MaskVg2000NDVI.tif",
                r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\MaskVg2001NDVI.tif",
                r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\MaskVg2002NDVI.tif",
                r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\_2003NDVI0811vgGEMUnion.tif",
                r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\_2004_NDVI_SG_0809new12vgGEMUnion.tif",
            ],
            "out_filename": "MaskVg_2000-2004_Months_3-11_Interannual_STD_NDVI.tif"
        },
        # # --- Task 3: 2005-2009 NDMI ---
        # {
        #     "name": "2005-2009_NDMI",
        #     "file_list": [
        #         r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\MaskVg2005NDMI.tif",
        #         r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\MaskVg2006NDMI.tif",
        #         r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\MaskVg2007NDMI.tif",
        #         r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\MaskVg2008NDMI.tif",
        #         r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\MaskVg2009NDMI.tif",
        #     ],
        #     "out_filename": "MaskVg_2005-2009_Months_3-11_Interannual_STD_NDMI.tif"
        # },
        # # --- Task 4: 2005-2009 NDVI ---
        # {
        #     "name": "2005-2009_NDVI",
        #     "file_list": [
        #         r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\MaskVg2005NDVI.tif",
        #         r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\MaskVg2006NDVI.tif",
        #         r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\MaskVg2007NDVI.tif",
        #         r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\MaskVg2008NDVI.tif",
        #         r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\MaskVg2009NDVI.tif",
        #     ],
        #     "out_filename": "MaskVg_2005-2009_Months_3-11_Interannual_STD_NDVI.tif"
        # },
        # # --- Task 5: 2010-2014 NDMI ---
        # {
        #     "name": "2010-2014_NDMI",
        #     "file_list": [
        #         r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\MaskVg2010NDMI.tif",
        #         r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\MaskVg2011NDMI.tif",
        #         r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\MaskVg2012NDMI.tif",
        #         r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\MaskVg2013NDMI.tif",
        #         r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\MaskVg2014NDMI.tif",
        #     ],
        #     "out_filename": "MaskVg_2010-2014_Months_3-11_Interannual_STD_NDMI.tif"
        # },
        # # --- Task 6: 2010-2014 NDVI ---
        # {
        #     "name": "2010-2014_NDVI",
        #     "file_list": [
        #         r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\MaskVg2010NDVI.tif",
        #         r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\MaskVg2011NDVI.tif",
        #         r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\MaskVg2012NDVI.tif",
        #         r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\MaskVg2013NDVI.tif",
        #         r"I:\GDEs_Data\SG_result_lt15\Basins_Output\yiliDissInts\125\Masked_OutputGEMunion125\MaskVg2014NDVI.tif",
        #     ],
        #     "out_filename": "MaskVg_2010-2014_Months_3-11_Interannual_STD_NDVI.tif"
        # }
    ]

    # ================= 3. Batch Execution Loop =================
    total_tasks = len(batch_tasks)
    print("==================================================")
    print(f"Starting batch calculation. Total tasks to process: {total_tasks}")
    print("==================================================\n")

    for idx, task in enumerate(batch_tasks, 1):
        print(f"[{idx}/{total_tasks}] Processing task: [{task['name']}] ...")
        out_filepath = os.path.join(out_dir, task['out_filename'])

        try:
            calculate_interannual_std(
                file_list=task['file_list'],
                month_bands=month_bands,
                out_file=out_filepath,
                block_size=block_size,
                nodata_val=nodata_val
            )
        except Exception as e:
            print(f"❌ Task [{task['name']}] failed! Error message: {e}\n")
            # Uncomment the line below if you want the script to terminate immediately upon an error
            # raise e

    print("🎉 All batch processing tasks have completed successfully!")