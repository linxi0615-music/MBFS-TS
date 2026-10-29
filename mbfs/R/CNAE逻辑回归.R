# ==============================================================================
# CNAE-9数据集的特征选择实验 (R语言版本)
# 修复 glmnet 收敛问题与多分类矩阵提取问题
# ==============================================================================

library(glmnet)
library(rpart)

# ==============================================================================
# 1. 数据加载与预处理函数
# ==============================================================================
load_cnae_data <- function(file_path) {
  cat("\n正在加载CNAE-9数据集...\n")
  
  if (!file.exists(file_path)) {
    stop(paste("找不到文件:", file_path))
  }
  
  data <- read.csv(file_path, header = FALSE)
  
  Y <- as.factor(data[, 1])
  X <- as.matrix(data[, -1])
  
  X[is.na(X)] <- 0
  X_scaled <- scale(X)
  X_scaled[is.na(X_scaled)] <- 0
  
  cat(sprintf("数据加载完毕: %d 样本, %d 特征\n", nrow(X_scaled), ncol(X_scaled)))
  
  return(list(X = X_scaled, y = Y))
}

# ==============================================================================
# 2. 分层 K 折交叉验证评估函数
# ==============================================================================
stratified_kfold <- function(y, n_folds = 5, seed = 42) {
  set.seed(seed)
  folds <- vector("list", n_folds) 
  
  classes <- unique(y)
  
  for (cls in classes) {
    idx <- which(y == cls)
    idx <- sample(idx)
    for (i in 1:length(idx)) {
      fold_id <- (i - 1) %% n_folds + 1
      folds[[fold_id]] <- c(folds[[fold_id]], idx[i])
    }
  }
  return(folds)
}

evaluate_with_cv <- function(X, y, selected_features = NULL, n_folds = 5, seed = 42) {
  if (!is.null(selected_features) && length(selected_features) > 0) {
    X_sel <- X[, selected_features, drop = FALSE]
  } else {
    X_sel <- X
  }
  
  df <- as.data.frame(X_sel)
  df$target <- y
  
  folds <- stratified_kfold(y, n_folds, seed)
  accuracies <- numeric(n_folds)
  
  for (i in 1:n_folds) {
    val_idx <- folds[[i]]
    train_idx <- setdiff(1:nrow(X_sel), val_idx)
    
    train_data <- df[train_idx, ]
    val_data <- df[val_idx, ]
    
    dt <- rpart(target ~ ., 
                data = train_data, 
                method = "class",
                control = rpart.control(maxdepth = 5, minsplit = 5, minbucket = 2))
    
    preds <- predict(dt, val_data, type = "class")
    
    accuracies[i] <- mean(preds == val_data$target)
  }
  
  return(list(mean = mean(accuracies), std = sd(accuracies)))
}

# ==============================================================================
# 3. L1-Logistic 特征排名函数 (修复版)
# ==============================================================================
logistic_feature_ranking <- function(X, y, seed = 42) {
  start_time <- Sys.time()
  set.seed(seed)
  
  family_type <- if (length(unique(y)) > 2) "multinomial" else "binomial"
  
  # 【修复点1】：不强制指定 lambda，让 glmnet 自动生成路径（避免不收敛）
  # 同时设置 standardize = FALSE，因为前面已经对 X 做过了 scale
  fit <- glmnet(X, y, family = family_type, alpha = 1, standardize = FALSE, maxit = 1e6)
  
  # 【修复点2】：使用 s = 0.05 提取目标 lambda 的系数
  coefs <- coef(fit, s = 0.05)
  
  if (family_type == "multinomial") {
    # 【修复点3】：针对多分类，稳健地提取每个类别的系数
    # coefs 是一个 list，包含了每个类别的系数矩阵
    importance_matrix <- sapply(1:length(coefs), function(k) {
      as.matrix(coefs[[k]])[-1, 1] # 去掉截距项
    })
    
    # 如果是单列，将其转为矩阵
    if (is.null(dim(importance_matrix))) {
      importance_matrix <- matrix(importance_matrix, ncol = 1)
    }
    
    # 取所有类别中绝对值最大的作为该特征的重要性
    importance <- apply(abs(importance_matrix), 1, max)
    
  } else {
    importance <- abs(as.matrix(coefs)[-1, 1])
  }
  
  feature_ranking <- order(importance, decreasing = TRUE)
  
  elapsed_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  return(list(ranking = feature_ranking, time = elapsed_time))
}

# ==============================================================================
# 4. 实验逻辑
# ==============================================================================
run_single_experiment <- function(X, y, run_id, feature_counts, n_folds = 5) {
  cat(sprintf("\n%s\n开始第 %d 次实验\n%s\n", strrep("=", 80), run_id, strrep("=", 80)))
  
  current_seed <- 42 + run_id * 10
  
  baseline <- evaluate_with_cv(X, y, selected_features = NULL, n_folds = n_folds, seed = current_seed)
  cat(sprintf("基准准确率 (All Features): %.6f (± %.6f)\n", baseline$mean, baseline$std))
  
  cat("\n处理方法: Logistic\n")
  lr_res <- logistic_feature_ranking(X, y, seed = current_seed)
  feature_ranking <- lr_res$ranking
  cat(sprintf("  特征排名计算完成，耗时: %.2f秒\n", lr_res$time))
  
  method_means <- numeric(length(feature_counts))
  method_stds <- numeric(length(feature_counts))
  
  for (i in seq_along(feature_counts)) {
    n_feat <- feature_counts[i]
    if (n_feat > length(feature_ranking)) next
    
    cat(sprintf("  特征数量: %d\n", n_feat))
    selected_idx <- feature_ranking[1:n_feat]
    
    cv_res <- evaluate_with_cv(X, y, selected_idx, n_folds = n_folds, seed = current_seed)
    cat(sprintf("    准确率: %.6f (± %.6f)\n", cv_res$mean, cv_res$std))
    
    method_means[i] <- cv_res$mean
    method_stds[i] <- cv_res$std
  }
  
  return(list(
    baseline_mean = baseline$mean,
    baseline_std = baseline$std,
    logistic_mean = method_means,
    logistic_std = method_stds
  ))
}

run_multiple_experiments <- function(X, y, feature_counts, n_runs = 5, n_folds = 5) {
  results <- list(
    baseline_means = numeric(n_runs),
    baseline_stds = numeric(n_runs),
    logistic_means = matrix(NA, nrow = n_runs, ncol = length(feature_counts)),
    logistic_stds = matrix(NA, nrow = n_runs, ncol = length(feature_counts))
  )
  
  for (run in 1:n_runs) {
    cat(sprintf("\n%s\n实验进度: %d/%d\n%s\n", strrep("#", 80), run, n_runs, strrep("#", 80)))
    
    res <- run_single_experiment(X, y, run, feature_counts, n_folds)
    
    results$baseline_means[run] <- res$baseline_mean
    results$baseline_stds[run] <- res$baseline_std
    results$logistic_means[run, ] <- res$logistic_mean
    results$logistic_stds[run, ] <- res$logistic_std
  }
  
  return(results)
}

# ==============================================================================
# 5. 结果打印与保存
# ==============================================================================
print_and_save_results <- function(results, feature_counts, dataset_name = "CNAE-9") {
  n_runs <- length(results$baseline_means)
  
  cat(sprintf("\n%s\n%s 数据集 - %d 次实验汇总结果\n%s\n", strrep("=", 80), dataset_name, n_runs, strrep("=", 80)))
  
  summary_df <- data.frame(
    n_features = feature_counts,
    mean_accuracy = colMeans(results$logistic_means, na.rm = TRUE),
    std_accuracy = colMeans(results$logistic_stds, na.rm = TRUE)
  )
  
  cat(sprintf("\n%-10s %-25s\n", "特征数", "Logistic (Mean ± SD)"))
  cat(strrep("-", 40), "\n")
  for (i in 1:nrow(summary_df)) {
    cat(sprintf("%-10d %.6f ± %.6f\n", 
                summary_df$n_features[i], 
                summary_df$mean_accuracy[i], 
                summary_df$std_accuracy[i]))
  }
  
  baseline_mean <- mean(results$baseline_means)
  baseline_std <- mean(results$baseline_stds)
  cat(sprintf("\n基准准确率 (所有特征): %.6f ± %.6f\n", baseline_mean, baseline_std))
  
  best_idx <- which.max(summary_df$mean_accuracy)
  best_n <- summary_df$n_features[best_idx]
  best_acc <- summary_df$mean_accuracy[best_idx]
  best_std <- summary_df$std_accuracy[best_idx]
  improvement <- (best_acc - baseline_mean) * 100
  
  cat(sprintf("\n%s\n最佳性能\n%s\n", strrep("=", 80), strrep("=", 80)))
  cat(sprintf("方法: Logistic | 最佳特征数: %d | 最佳准确率: %.6f ± %.6f | 提升: %+.2f%%\n", 
              best_n, best_acc, best_std, improvement))
  
  summary_df$method <- "Logistic"
  summary_df <- summary_df[, c("method", "n_features", "mean_accuracy", "std_accuracy")]
  
  baseline_df <- data.frame(
    method = "Baseline",
    n_features = "All",
    mean_accuracy = baseline_mean,
    std_accuracy = baseline_std
  )
  summary_df <- rbind(baseline_df, summary_df)
  
  write.csv(summary_df, paste0(dataset_name, "_summary.csv"), row.names = FALSE)
  cat(sprintf("\n汇总结果已保存到: %s_summary.csv\n", dataset_name))
}

# ==============================================================================
# 6. 主程序
# ==============================================================================
main <- function() {
  cat(strrep("=", 80), "\n")
  cat("CNAE-9数据集 - 特征选择实验 (仅 L1-Logistic)\n")
  cat(strrep("=", 80), "\n")
  
  file_path <- 'C:/Users/锡/Downloads/cnae+9/CNAE-9.csv'
  
  if (!file.exists(file_path)) {
    alt_paths <- c("CNAE-9.csv", "./CNAE-9.csv", "../CNAE-9.csv")
    for (p in alt_paths) {
      if (file.exists(p)) {
        file_path <- p
        break
      }
    }
  }
  
  if (!file.exists(file_path)) {
    stop("无法找到 CNAE-9 数据集文件，请检查路径。")
  }
  
  data <- load_cnae_data(file_path)
  X <- data$X
  y <- data$y
  
  max_features <- min(30, ncol(X))
  feature_counts <- 1:max_features
  n_runs <- 5
  n_folds <- 5
  
  cat(sprintf("\n实验参数:\n  特征数量测试范围: 1 - %d\n  重复实验次数: %d\n  交叉验证折数: %d\n", 
              max_features, n_runs, n_folds))
  
  results <- run_multiple_experiments(X, y, feature_counts, n_runs, n_folds)
  
  print_and_save_results(results, feature_counts, "CNAE-9")
  
  cat("\n", strrep("=", 80), "\n")
  cat("实验完成！\n")
  cat(strrep("=", 80), "\n")
}

main()