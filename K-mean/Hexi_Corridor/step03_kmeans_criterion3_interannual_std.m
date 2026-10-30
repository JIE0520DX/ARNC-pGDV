%% =========================================================================
% Script Name: step03_kmeans_criterion3_interannual_std_hexi.m
% Description: K-means clustering for Criterion 3 (Interannual fluctuation magnitude 
%              calculated across 5-year to ensure long-term stability).
% Study Area:  Arid Region of Northwest China (ARNC) - Hexi Corridor (Basin III)
% Periods:     2000-2025
% Author:      Jie Zhou (zhenleiyang@ms.xjb.ac.cn)
% Reference:   Zhou et al., Scientific Data (Nature Portfolio), 2026.
% Environment: MATLAB R2025b
%% =========================================================================

%% ================= User Configuration Zone =================
% 1. Input directory: folder containing all target standard deviation GeoTIFF files
input_dir = 'I:\GDEs_Data\SG_result\Basins_Output\hexiDissInts\hexi0726UnionGEM\NDVI_NDMI\hexiDissInts\STD_python';

% 2. Output subdirectory name (automatically created under input_dir)
output_subdir = 'MC';

% 3. File pattern to match (processes GeoTIFF files starting with MaskVg)
file_pattern = 'MaskVg*.tif';

% 4. K-means clustering parameters
num_cluster = 5;       % Number of clusters
max_iter = 1000;       % Maximum number of iterations
% ===================================================================

%% Initialization and File Retrieval
% Construct full output directory path
output_dir = fullfile(input_dir, output_subdir);
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Retrieve list of all matching GeoTIFF files
file_list = dir(fullfile(input_dir, file_pattern));
if isempty(file_list)
    error('No matching GeoTIFF files found in the specified directory. Please check the path and file pattern.');
end
fprintf('Found %d raster file(s) to process, starting batch clustering...\n', length(file_list));

%% Batch Processing Workflow
for i = 1:length(file_list)
    % Full path of current file
    input_file = fullfile(input_dir, file_list(i).name);
    fprintf('[%d/%d] Processing: %s\n', i, length(file_list), file_list(i).name);
    
    % 1. Read raster data and spatial reference
    [std_data, R] = geotiffread(input_file);
    
    % 2. Cast data type to double
    std_data = double(std_data);
    
    % 3. Filter valid pixels (exclude NoData values; original NoData is -32768)
    valid_idx = find(std_data > -100);
    valid_data = std_data(valid_idx);
    
    % 4. K-means clustering
    cluster_result = kmeans(valid_data, num_cluster, ...
        'MaxIter', max_iter, 'Start', 'sample');
    
    % 5. Reconstruct result matrix with original raster dimensions in int8 format
    result = zeros(size(std_data, 1), size(std_data, 2), 'int8');
    result(valid_idx) = cluster_result;
    
    % 6. Construct output file name and export georeferenced raster
    [~, name_base, ~] = fileparts(file_list(i).name); % Extract base file name without extension
    out_file = fullfile(output_dir, [name_base, '_MC.tif']);
    geotiffwrite(out_file, result, R);
    
    fprintf('  Saved: %s\n', [name_base, '_MC.tif']);
end

fprintf('\nAll files processed successfully! Results saved to: %s\n', output_dir);