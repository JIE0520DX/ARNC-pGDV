%% =========================================================================
% Script Name: step02_kmeans_criterion2_seasonal_std_hexi.m
% Description: K-means clustering for Criterion 2 (Seasonal fluctuation magnitude 
%              [standard deviation of NDVI and NDMI from dry-season end to wet-season end]).
% Study Area:  Arid Region of Northwest China (ARNC) - Hexi Corridor (Basin III)
% Periods:     2000-2025
% Author:      Jie Zhou (zhenleiyang@ms.xjb.ac.cn)
% Reference:   Zhou et al., Scientific Data (Nature Portfolio), 2026.
% Environment: MATLAB R2025b
%% =========================================================================
clear; clc; close all;

% ==============================================================
% =================== User Configuration Zone ==================
% ==============================================================
% 1. Root directory: parent directory of all basin folders
root_dir = 'I:\GDEs_Data\SG_result\Basins_Output\hexiDissInts\hexi0726UnionGEM\NDVI_NDMI';

% 2. Basin configuration
basin_config = [];
basin_config(end+1).name  = 'hexiDissInts';   % Basin folder name
basin_config(end).bands   = 4:10;                 % Band range for standard deviation calculation: months 4-10

% 3. General clustering parameters
num_clusters = 5;          % Number of clusters
max_iter     = 1000;       % Maximum number of iterations
file_pattern = 'MaskVg*.tif';  % File pattern to match

% Valid pixel condition
valid_cond = @(x) x > -10000 & x ~= -32768;
nodata_int8 = int8(-128);  % NoData value for int8 raster output to avoid confusion with classes 1-5

% ==============================================================
% ===================== Processing Workflow ====================
% ==============================================================
fprintf('Configured %d basin(s), starting batch processing [STD + K-Means clustering]...\n', length(basin_config));

for i = 1:length(basin_config)
    basin_name = basin_config(i).name;
    band_list  = basin_config(i).bands;
    basin_path = fullfile(root_dir, basin_name);
    
    fprintf('\n========================================\n');
    fprintf('[%d/%d] Basin: %s | Target Bands: %s | Clusters: %d\n', ...
        i, length(basin_config), basin_name, num2str(band_list), num_clusters);
    fprintf('========================================\n');
    
    % Check if basin directory exists
    if ~exist(basin_path, 'dir')
        fprintf('  ✗ Error: Basin directory not found, skipped\n');
        continue;
    end
    
    % Create output directory
    output_dir = fullfile(basin_path, 'MC_MultiMonYearStd_Cluster');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % Match all target GeoTIFF files under current basin
    tif_list = dir(fullfile(basin_path, file_pattern));
    if isempty(tif_list)
        fprintf('  Warning: No matching GeoTIFF files found, skipped\n');
        continue;
    end
    fprintf('  Found %d raster file(s) to process\n', length(tif_list));
    
    % Iterate over each raster file
    for j = 1:length(tif_list)
        tif_name = tif_list(j).name;
        tif_path = fullfile(basin_path, tif_name);
        [~, file_prefix, ~] = fileparts(tif_name);
        fprintf('\n  -> Processing file: %s\n', tif_name);
        
        try
            % Read multi-band raster and spatial reference
            [value_multi, R] = geotiffread(tif_path);
            [h, w, total_bands] = size(value_multi);
            fprintf('    Total raster bands: %d\n', total_bands);
            
            % Check validity of specified band range
            if max(band_list) > total_bands
                fprintf('    Specified bands exceed total band count, skipped\n');
                continue;
            end
            
            % ========== Step 1: Compute pixel-wise standard deviation for months 4-10 ==========
            data = double(value_multi(:, :, band_list));
            % Replace invalid values with NaN, automatically ignored in standard deviation
            data(~valid_cond(data)) = NaN;
            % Compute standard deviation along the 3rd dimension (bands/months)
            std_arr = nanstd(data, 0, 3);
            
            % Filter valid standard deviation pixels (non-NaN)
            valid_mask = ~isnan(std_arr);
            valid_std_vals = std_arr(valid_mask);
            if isempty(valid_std_vals)
                fprintf('    No valid STD pixels found, skipped\n');
                continue;
            end
            fprintf('    Valid pixel count: %d\n', length(valid_std_vals));
            
            % ========== Step 2: K-Means clustering on standard deviation values ==========
            cluster_idx = kmeans(valid_std_vals, num_clusters, ...
                'MaxIter', max_iter, ...
                'Start', 'sample', ...
                'Display', 'off');
            
            % Write cluster indices back to raster grid with int8 data type, invalid pixels set to NoData
            result = repmat(nodata_int8, h, w);
            result(valid_mask) = int8(cluster_idx);
            
            % Output file name format
            output_name = sprintf('%s_%s_Months4-10_STD_Cluster%d_MC.tif', ...
                basin_name, file_prefix, num_clusters);
            output_path = fullfile(output_dir, output_name);
            
            % Export georeferenced raster result
            geotiffwrite(output_path, result, R);
            fprintf('    Clustering complete | Exported: %s\n', output_name);
        catch ME
            fprintf('    ✗ Processing failed: %s\n', ME.message);
            continue;
        end
    end
end

fprintf('\nAll basins processed successfully! Results saved in the respective MC_MultiMonYearStd_Cluster folders.\n');