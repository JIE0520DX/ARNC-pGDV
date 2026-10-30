function [label, center] = isodata(X, init_centers, K, theta_N, theta_S, theta_C, L, max_iter)
% ISODATA 聚类算法
% 输入:
%   X            - n x d 数据矩阵
%   init_centers - 初始聚类中心 (k0 x d)
%   K            - 期望聚类数
%   theta_N      - 每类最少样本数阈值
%   theta_S      - 标准差阈值（用于分裂）
%   theta_C      - 类间最小距离阈值（用于合并）
%   L            - 每次迭代允许合并的最大对数
%   max_iter     - 最大迭代次数
% 输出:
%   label        - n x 1 类别标签
%   center       - 最终聚类中心
% 统一转为 single 类型（避免警告，节省内存）
    X = single(X);
    init_centers = single(init_centers);
    n = size(X, 1);
    centers = init_centers;
    k = size(centers, 1);
    iter = 0;
    prev_centers = [];   % 初始为空

    while iter < max_iter
        % ---------- 1. 分配每个样本到最近的聚类中心 ----------
        dist = pdist2(X, centers);
        [~, label] = min(dist, [], 2);
        
        % ---------- 2. 统计各类样本数，删除样本数少于 theta_N 的类 ----------
        count = histcounts(label, 1:(k+1));
        valid = count >= theta_N;
        if ~all(valid)
            centers = centers(valid, :);
            k = size(centers, 1);
            % 重新分配标签
            if k > 0
                dist = pdist2(X, centers);
                [~, label] = min(dist, [], 2);
                count = histcounts(label, 1:(k+1));
            else
                error('所有类都被删除，请减小 theta_N');
            end
        end
        
        % ---------- 3. 更新聚类中心 ----------
        new_centers = zeros(k, size(X,2));
        for i = 1:k
            members = X(label == i, :);
            if ~isempty(members)
                new_centers(i,:) = mean(members, 1);
            else
                new_centers(i,:) = centers(i,:);   % 防御
            end
        end
        centers = new_centers;
        
        % ---------- 4. 计算类内平均距离和整体平均距离 ----------
        avg_dist_within = zeros(k,1);
        for i = 1:k
            members = X(label == i, :);
            if size(members,1) > 0
                avg_dist_within(i) = mean(pdist2(members, centers(i,:)));
            else
                avg_dist_within(i) = 0;
            end
        end
        overall_avg_dist = mean(avg_dist_within);
        
        % ---------- 5. 分裂操作 ----------
        if k <= K/2 || mod(iter,2) == 0   % 原算法常用条件：类数太少或偶次迭代
            split_indices = [];
            for i = 1:k
                members = X(label == i, :);
                if size(members,1) >= 2*theta_N
                    sigma = std(members, 0, 1);
                    max_sigma = max(sigma);
                    if max_sigma > theta_S
                        % 找到标准差最大的维度
                        [~, dim] = max(sigma);
                        % 分裂条件：类内平均距离 > 整体平均距离
                        if avg_dist_within(i) > overall_avg_dist
                            split_indices = [split_indices, i];
                        end
                    end
                end
            end
            % 执行分裂
            for idx = split_indices
                members = X(label == idx, :);
                [~, dim] = max(std(members,0,1));
                center_old = centers(idx,:);
                alpha = 0.5 * std(members(:,dim));
                if alpha == 0
                    alpha = 0.01;   % 防止零偏移
                end
                center1 = center_old;
                center1(dim) = center1(dim) + alpha;
                center2 = center_old;
                center2(dim) = center2(dim) - alpha;
                centers(idx,:) = center1;
                centers = [centers; center2];
            end
            k = size(centers,1);
            % 重新分配标签
            dist = pdist2(X, centers);
            [~, label] = min(dist, [], 2);
            count = histcounts(label, 1:(k+1));
        end
        
        % ---------- 6. 合并操作 ----------
        if k > 2*K || mod(iter,2) == 1   % 类数太多或奇次迭代
            center_dist = pdist2(centers, centers);
            center_dist = center_dist + diag(inf(k,1));
            pairs = [];
            for i = 1:k
                for j = i+1:k
                    if center_dist(i,j) < theta_C
                        pairs = [pairs; i, j, center_dist(i,j)];
                    end
                end
            end
            if ~isempty(pairs)
                pairs = sortrows(pairs, 3);
                used = false(k,1);
                merge_count = 0;
                for p = 1:size(pairs,1)
                    i = pairs(p,1);
                    j = pairs(p,2);
                    if ~used(i) && ~used(j) && merge_count < L
                        ci = count(i);
                        cj = count(j);
                        new_center = (ci*centers(i,:) + cj*centers(j,:)) / (ci + cj);
                        centers(i,:) = new_center;
                        used(j) = true;
                        merge_count = merge_count + 1;
                    end
                end
                centers = centers(~used, :);
                k = size(centers,1);
                dist = pdist2(X, centers);
                [~, label] = min(dist, [], 2);
                count = histcounts(label, 1:(k+1));
            end
        end
        
        % ---------- 7. 收敛检查（修正维度不一致问题） ----------
        if iter > 0
            % 只有当前后两次迭代的聚类中心数目相同且中心值相同时，才认为收敛
            if size(prev_centers,1) == size(centers,1) && all(all(abs(prev_centers - centers) < 1e-6))
                break;
            end
        end
        prev_centers = centers;   % 保存本次迭代结果
        iter = iter + 1;
        
        % 可选：显示进度
        % fprintf('Iter %d: number of clusters = %d\n', iter, k);
    end
    center = centers;   % <<--- 添加这一行
end