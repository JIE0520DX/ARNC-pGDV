%% =========================================================================
% Script Name: step02_kmeans_criterion2_seasonal_std_tarim.m
% Description: K-means clustering for Criterion 2 (Seasonal fluctuation magnitude 
%              [standard deviation of NDVI and NDMI from dry-season end to wet-season end]).
% Study Area:  Arid Region of Northwest China (ARNC) - Tarim Basin (Basin IV)
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

% 2. Basin configuration
basin_config = [];
basin_config(end+1).name  = 'tarimDissInts';   % Basin folder name
basin_config(end).bands   = 4:10;              % Band range for standard deviation calculation: months 4-10

% 3. General clustering and processing parameters
num_clusters = 5;          % Number of clusters
max_iter     = 1000;       % Maximum number of iterations
file_pattern = 'MaskVg*.tif';  % File pattern to match
block_size   = 1000;       % Block size (pixels); reduce if memory is constrained (e.g., 1000)

% Valid pixel condition
valid_cond = @(x) x > -10000 & x ~= -32768;
nodata_int8 = int8(-128);  % NoData value for int8 raster output

% ==============================================================
% ===================== Processing Workflow ====================
% ==============================================================
fprintf('Configured %d basin(s), starting batch processing [Block-wise STD + K-Means clustering]...\n', length(basin_config));

for i = 1:length(basin_config)
    basin_name = basin_config(i).name;
    band_list  = basin_config(i).bands;
    basin_path = fullfile(root_dir, basin_name);
    
    fprintf('\n========================================\n');
    fprintf('[%d/%d] Basin: %s | Target Bands: %s | Clusters: %d\n', ...
        i, length(basin_config), basin_name, num2str(band_list), num_clusters);
    fprintf('========================================\n');
    
    if ~exist(basin_path, 'dir')
        fprintf('  ✗ Error: Basin directory not found, skipped\n');
        continue;
    end
    
    output_dir = fullfile(basin_path, 'MC_MultiMonYearStd_Cluster');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir); 
    end
    
    tif_list = dir(fullfile(basin_path, file_pattern));
    if isempty(tif_list)
        fprintf('  Warning: No matching GeoTIFF files found, skipped\n');
        continue;
    end
    fprintf('  Found %d raster file(s) to process\n', length(tif_list));
    
    for j = 1:length(tif_list)
        tif_name = tif_list(j).name;
        tif_path = fullfile(basin_path, tif_name);
        [~, file_prefix, ~] = fileparts(tif_name);
        fprintf('\n  -> Processing file: %s\n', tif_name);
        
        try
            % Retrieve spatial reference and raster dimensions without loading full array
            info = geotiffinfo(tif_path);
            R = info.SpatialRef;
            h = info.Height; 
            w = info.Width;
            
            im_info = imfinfo(tif_path);
            total_bands = im_info(1).SamplesPerPixel; % Retrieve band count
            fprintf('    Dimensions: %dx%d | Total Bands: %d\n', h, w, total_bands);
            if max(band_list) > total_bands
                fprintf('    Specified bands exceed total band count, skipped\n');
                continue;
            end
            
            % Block-wise read and calculate standard deviation
            std_arr = NaN(h, w, 'double'); % Preallocate 2D standard deviation matrix
            
            for r = 1:block_size:h
                for c = 1:block_size:w
                    % Calculate row and column extents of current block
                    r_end = min(r + block_size - 1, h);
                    c_end = min(c + block_size - 1, w);
                    
                    % Read pixel region of current block
                    block_data = imread(tif_path, 'PixelRegion', {[r, r_end], [c, c_end]});
                    
                    % Extract target bands, mask invalid pixels, and compute standard deviation
                    data = double(block_data(:, :, band_list));
                    data(~valid_cond(data)) = NaN;
                    
                    % Assign block results into preallocated matrix
                    std_arr(r:r_end, c:c_end) = nanstd(data, 0, 3);
                end
            end
            fprintf('    Block-wise STD calculation complete, starting clustering...\n');
            
            % Filter valid pixels for clustering
            valid_mask = ~isnan(std_arr);
            valid_std_vals = std_arr(valid_mask);
            if isempty(valid_std_vals)
                fprintf('    No valid STD pixels found, skipped\n');
                continue;
            end
            fprintf('    Valid pixel count: %d\n', length(valid_std_vals));
            
            % K-means clustering (Display off for batch execution)
            cluster_idx = kmeans(valid_std_vals, num_clusters, ...
                'MaxIter', max_iter, ...
                'Start', 'sample', ...
                'Display', 'off');
            
            % Write cluster indices back to raster grid
            result = repmat(nodata_int8, h, w);
            result(valid_mask) = int8(cluster_idx);
            
            % Output file name format: Basin_OriginalName_Months4-10_STD_ClusterX_MC.tif
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

fprintf('\nAll basins processed successfully!\n');