%% =========================================================================
% Script Name: step01_kmeans_criterion1_end_of_dry_season_yili.m
% Description: K-means clustering for Criterion 1 (Vegetation greenness [NDVI] 
%              and canopy wetness [NDMI] at the end of the dry season).
% Study Area:  Arid Region of Northwest China (ARNC) - Yili River Basin (Basin V)
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
root_dir = 'L:\GDEs_Data\GEM_NW2000_25Rec125Mos_VG\DryNDVI_NDMI';

% 2. Basin configuration: specify folder name and clustering bands for each basin
%    Add new basin: duplicate a basin_config(end+1) line and modify name/bands
basin_config = [];
% Example basin 1: Yili River Basin
basin_config(end+1).name  = 'yiliDissInts';   % Basin folder name
basin_config(end).bands   = [3];              % Band index to cluster, supports multiple bands e.g., [4,7,10]

% 3. General clustering parameters
num_clusters = 5;          % Number of clusters
max_iter     = 1000;       % Maximum number of iterations
file_pattern = 'MaskVg*.tif';  % File pattern to match

% Valid pixel condition
valid_cond = @(x) x > -10000 & x ~= -32768;

% ==============================================================
% ===================== Processing Workflow ====================
% ==============================================================
fprintf('Configured %d basin(s), starting batch processing...\n', length(basin_config));

for i = 1:length(basin_config)
    basin_name = basin_config(i).name;
    band_list  = basin_config(i).bands;
    basin_path = fullfile(root_dir, basin_name);
    
    fprintf('\n========================================\n');
    fprintf('[%d/%d] Basin: %s | Target Band(s): %s\n', ...
        i, length(basin_config), basin_name, num2str(band_list));
    fprintf('========================================\n');
    
    % Check if basin directory exists
    if ~exist(basin_path, 'dir')
        fprintf('  ✗ Error: Basin directory not found, skipped\n');
        continue;
    end
    
    % Create output directory MC
    output_dir = fullfile(basin_path, 'MC');
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
            total_bands = size(value_multi, 3);
            fprintf('    Total raster bands: %d\n', total_bands);
            
            % Iterate through all specified bands for this basin
            for b = band_list
                if b > total_bands
                    fprintf('    Band %d exceeds total band count, skipped\n', b);
                    continue;
                end
                fprintf('    Processing Band %d...', b);
                
                % Extract single band and cast to double
                value_band = double(value_multi(:, :, b));
                
                % Filter valid pixels
                valid_mask = valid_cond(value_band);
                valid_vals = value_band(valid_mask);
                if isempty(valid_vals)
                    fprintf('No valid pixels found, skipped\n');
                    continue;
                end
                
                % K-means clustering (Display off for batch execution)
                cluster_idx = kmeans(valid_vals, num_clusters, ...
                    'MaxIter', max_iter, ...
                    'Start', 'sample', ...
                    'Display', 'off');
                
                % Write results back to original raster grid with int8 data type
                result = zeros(size(value_band), 'int8');
                result(valid_mask) = int8(cluster_idx);
                
                % Output file name format: Basin_OriginalName_BandX_MC.tif
                output_name = sprintf('%s_%s_Band%d_MC.tif', basin_name, file_prefix, b);
                output_path = fullfile(output_dir, output_name);
                
                % Export georeferenced raster result
                geotiffwrite(output_path, result, R);
                fprintf('Done | Valid pixels: %d | Exported\n', length(valid_vals));
            end
        catch ME
            fprintf('    ✗ Processing failed: %s\n', ME.message);
            continue;
        end
    end
end

fprintf('\nAll basins processed successfully! Results saved in the respective MC folders.\n');