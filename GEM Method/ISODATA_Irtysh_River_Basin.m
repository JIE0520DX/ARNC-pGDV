% =========================================================================
% Script Name: ISODATA_Classification_MultiYear.m
% Description: Multi-year blockwise ISODATA clustering and vegetation/moisture
%              index analysis using multi-band GeoTIFF time series data.
% Study Area:  Arid Region of Northwest China (ARNC) - Irtysh River Basin (Basin I)
% Periods:     2000-2025
% Author:      Jie Zhou (zhenleiyang@ms.xjb.ac.cn)
% Reference:   Zhou et al., Scientific Data (Nature Portfolio), 2026.
% Environment: MATLAB R2025b
% =========================================================================

clear; clc;

% Define the range of years to process
years = 2000:2025;

for y = 1:length(years)
    current_year = years(y);
    year_str = num2str(current_year);
    
    fprintf('====================================================\n');
    fprintf('           Processing Year: %s\n', year_str);
    fprintf('====================================================\n');
    
    %% 1. Read multi-band GeoTIFF (dynamically generate file paths for the current year)
    tif_file_ndvi_last = ['I:\GDEs_Data\SG_result\Basins_Output\eerqis1DissInts\', 'MaskVg', year_str, 'NDVI.tif'];
    tif_file_ndvi_next = ['I:\GDEs_Data\SG_result\Basins_Output\eerqis1DissInts\', 'MaskVg', year_str, 'NDVI.tif'];
    tif_file_ndmi_last = ['I:\GDEs_Data\SG_result\Basins_Output\eerqis1DissInts\', 'MaskVg', year_str, 'NDMI.tif'];
    tif_file_ndmi_next = ['I:\GDEs_Data\SG_result\Basins_Output\eerqis1DissInts\', 'MaskVg', year_str, 'NDMI.tif'];
    
    % Check if input files exist for the current year to prevent execution interruption
    if ~exist(tif_file_ndvi_last, 'file')
        fprintf('Warning: Input file for year %s not found. Skipping to next year...\n', year_str);
        continue; 
    end
    
    % Retrieve basic spatial and dimension metadata from GeoTIFF
    info = geotiffinfo(tif_file_ndvi_last);
    rows = info.Height;  
    cols  = info.Width;  
    R = info.SpatialRef; 
    
    % Pre-allocate full memory space for the 4 required bands
    fprintf('Pre-allocating memory for the 4 bands in year %s...\n', year_str);
    NDVI_dry = zeros(rows, cols, 'int16');
    NDVI_wet = zeros(rows, cols, 'int16');
    NDWI_dry = zeros(rows, cols, 'int16');
    NDWI_wet = zeros(rows, cols, 'int16');
    
    %% Blockwise reading into global arrays
    block_size = 4000; 
    fprintf('Starting blockwise data reading. Total dimensions: %d x %d\n', rows, cols);
    for r = 1:block_size:rows
        r_end = min(r + block_size - 1, rows);
        for c = 1:block_size:cols
            c_end = min(c + block_size - 1, cols);
            
            % Read data blocks
            ndvi_block_last = imread(tif_file_ndvi_last, 'PixelRegion', {[r, r_end], [c, c_end]});
            ndvi_block_next = imread(tif_file_ndvi_next, 'PixelRegion', {[r, r_end], [c, c_end]});
            ndmi_block_last = imread(tif_file_ndmi_last, 'PixelRegion', {[r, r_end], [c, c_end]});
            ndmi_block_next = imread(tif_file_ndmi_next, 'PixelRegion', {[r, r_end], [c, c_end]});
            
            % Assign bands 4 and 11 into global matrices
            NDVI_dry(r:r_end, c:c_end) = ndvi_block_next(:,:,4);
            NDVI_wet(r:r_end, c:c_end) = ndvi_block_last(:,:,11);
            
            NDWI_dry(r:r_end, c:c_end) = ndmi_block_next(:,:,4);
            NDWI_wet(r:r_end, c:c_end) = ndmi_block_last(:,:,11);
        end
        fprintf('Loaded rows %d to %d...\n', r, r_end);
    end
    % Clear temporary block variables to free memory
    clear ndvi_block_last ndvi_block_next ndmi_block_last ndmi_block_next; 
    
    %% Compute valid pixel mask and indices
    fprintf('Computing valid pixel mask...\n');
    valid_mask = (NDVI_dry > -10000) & (NDVI_wet > -10000) & ...
                 (NDWI_dry > -10000) & (NDWI_wet > -10000);
    valid_idx = find(valid_mask);
    num_valid = length(valid_idx);
    fprintf('Total valid pixels: %d (%.2f%%)\n', num_valid, 100 * num_valid / numel(valid_mask));
    % Clear mask array
    clear valid_mask;
    
    %% 2. Random sampling of training points
    sample_ratio = 0.4;   
    sample_n = min(round(num_valid * sample_ratio), 2e6);  
    rng(42);  
    sample_idx = randsample(num_valid, sample_n);
    valid_linear = valid_idx(sample_idx);
    
    % Extract training features
    X_train = [NDVI_dry(valid_linear), NDVI_wet(valid_linear), ...
               NDWI_dry(valid_linear), NDWI_wet(valid_linear)];
    X_train = single(X_train) / 10000;   
    fprintf('Number of training samples: %d\n', size(X_train,1));
    
    %% 3. ISODATA clustering (Estimate cluster centers)
    K_expected = 13;
    theta_N = max(500, round(size(X_train,1) * 0.01));
    theta_S = 0.10;   % Standard deviation threshold for splitting
    theta_C = 0.20;   % Distance threshold for merging
    L = 15;
    max_iter = 1000;  
    k0 = 13;         
    init_idx = randperm(size(X_train,1), k0);
    init_centers = X_train(init_idx, :);   
    
    % Run ISODATA algorithm
    [label_train, centers] = isodata(X_train, init_centers, K_expected, ...
                                     theta_N, theta_S, theta_C, L, max_iter);
    fprintf('Clustering completed. Actual number of cluster centers: %d\n', size(centers,1));
    clear X_train; % Release training feature matrix
    
    %% 4. Full image classification by blocks
    label_img = zeros(rows, cols, 'uint8');
    block_rows = 5000;   
    block_cols = 5000;
    centers = single(centers);
    
    for i = 1:block_rows:rows
        r_end = min(i+block_rows-1, rows);
        for j = 1:block_cols:cols
            c_end = min(j+block_cols-1, cols);
            % Extract sub-blocks
            b_ndvi_dry = NDVI_dry(i:r_end, j:c_end);
            b_ndvi_wet = NDVI_wet(i:r_end, j:c_end);
            b_ndwi_dry = NDWI_dry(i:r_end, j:c_end);
            b_ndwi_wet = NDWI_wet(i:r_end, j:c_end);
            
            % Sub-block valid mask
            block_valid = (b_ndvi_dry > -10000) & (b_ndvi_wet > -10000) & ...
                          (b_ndwi_dry > -10000) & (b_ndwi_wet > -10000);
                          
            % Convert to single and scale to natural value range
            X_block = single([b_ndvi_dry(:), b_ndvi_wet(:), b_ndwi_dry(:), b_ndwi_wet(:)]) / 10000;
            n_block = size(X_block,1);
            
            % Batch distance calculation to avoid out-of-memory errors
            batch_size = 50000;
            label_block = zeros(n_block, 1, 'uint8');
            for start_idx = 1:batch_size:n_block
                end_idx = min(start_idx+batch_size-1, n_block);
                X_batch = X_block(start_idx:end_idx, :);
                dist = pdist2(X_batch, centers);   
                [~, batch_labels] = min(dist, [], 2);
                label_block(start_idx:end_idx) = uint8(batch_labels);
            end
            % Set invalid pixels to 0
            label_block(~block_valid(:)) = 0;
            label_img(i:r_end, j:c_end) = reshape(label_block, r_end-i+1, c_end-j+1);
        end
        fprintf('Completed rows %d / %d\n', r_end, rows);
    end
    
    %% 5. Export classification raster (dynamically named output GeoTIFF)
    output_file = ['eerqisGEM', year_str, 'ISODATA_22classes0629gt_10000_1.tif'];
    geotiffwrite(output_file, label_img, R, 'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);
    fprintf('Classification result for year %s saved to: %s\n', year_str, output_file);
    
    %% 6. Calculate mean spectral/moisture indices per cluster
    num_classes = size(centers, 1);  
    valuesum = zeros(num_classes, 5);
    for k = 1:num_classes
        suoyin = find(label_img == k);
        if ~isempty(suoyin)
            valuesum(k, 1) = k;
            valuesum(k, 2) = mean(double(NDVI_dry(suoyin)));
            valuesum(k, 3) = mean(double(NDVI_wet(suoyin)));
            valuesum(k, 4) = mean(double(NDWI_dry(suoyin)));
            valuesum(k, 5) = mean(double(NDWI_wet(suoyin)));
        end
    end
    valuesum(:, 2:end) = valuesum(:, 2:end) ./ 10000;
    
    % Save summary table to CSV (dynamically named output CSV)
    csv_file = ['eerqisGEM', year_str, 'cluster_metrics_0629gt_10000_1.csv'];
    csvwrite(csv_file, valuesum);
    fprintf('Cluster statistics for year %s saved to: %s\n', year_str, csv_file);
    
    % Diagnostic scatter plot
    % figure; % Uncomment to keep individual figure windows per year
    scatter(valuesum(:, 2), valuesum(:, 3), 'filled');
    xlabel('NDVI\_dry');
    ylabel('NDVI\_wet');
    title(sprintf('%s Cluster Mean NDVI (Dry Season vs. Wet Season)', year_str));
    drawnow; % Refresh plot display for current year
    
    %% 7. End of loop: Clear large variables to free memory for the next iteration
    clear NDVI_dry NDVI_wet NDWI_dry NDWI_wet label_img valid_idx valid_linear centers;
    fprintf('Data processing for year %s completed.\n\n', year_str);
    
end % End of year loop

fprintf('All years have been successfully processed.\n');