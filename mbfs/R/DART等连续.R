generate_data1 <- function(n = 3000, p = 300, n_signal = 5, noise_sd = 1) {
  sigma=1
  X <- matrix(runif(n * p), n, p)
  Y <- 10 * sin(pi * X[, 1] * X[, 2]) +
    20 * (X[, 3] - 0.5)^2 +
    10 * X[, 4] +
    5 * X[, 5] +
    rnorm(n, 0, sigma)
  return(list(X = X, Y = Y, signal_vars = 1:5))
}

generate_data <- function(n = 3000, p = 300, n_signal = 5, noise_sd = 2) {
  e <- rnorm(n)
  X <- matrix(0, n, p)
  for (i in 1:n) {
    for (j in 1:p) {
      z_ij <- rnorm(1)
      X[i, j] <- (e[i] + z_ij) / 2
    }
  }
  X <- pnorm(X)
  Y <- numeric(n)
  for (i in 1:n) {
    f0 <- (10 * X[i, 2]) / (1 + X[i, 1]^2) + 5 * sin(X[i, 3] * X[i, 4]) + 2 * X[i, 5]
    #Y[i] <- f0 + rnorm(1, 0, sqrt(0.5))
    Y[i] <- f0 + rnorm(1, 0, noise_sd)
  }
  signal_vars <- 1:5
  return(list(X = X, Y = Y, signal_vars = 1:5))
}
library(BART)
library(randomForest)
library(glmnet)
library(SSLASSO)
library(SIS)
library(horseshoe)
library(caret)    # 用于 rfe
# library(e1071) # 若 caret 报错需要 e1071，自动依赖
evaluate_selection2 <- function(prob, true_idx, topq = NULL, threshold = 0.5) {
  # 向后兼容：如果topq为NULL，使用threshold
  if (is.null(topq)) {
    sel <- as.numeric(prob > threshold)
  } else {
    sel <- rep(0, length(prob))
    top_indices <- order(prob, decreasing = TRUE)[1:min(topq, length(prob))]
    sel[top_indices] <- 1
  }
  
  truth <- rep(0, length(prob)); truth[true_idx] <- 1
  
  TP <- sum(sel == 1 & truth == 1)
  FP <- sum(sel == 1 & truth == 0)
  FN <- sum(sel == 0 & truth == 1)
  
  Precision <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)
  Recall <- ifelse((TP + FN) > 0, TP / (TP + FN), 0)
  F1 <- ifelse((Precision + Recall) > 0, 2 * Precision * Recall / (Precision + Recall), 0)
  FDP <- ifelse((TP + FP) > 0, FP / (TP + FP), 0)
  Hamming <- sum(abs(sel - truth))
  
  data.frame(
    FDP = FDP,
    Precision = Precision,
    Recall = Recall,
    F1 = F1,
    Hamming = Hamming,
    Time = NA
  )
}

evaluate_selection <- function(prob, true_idx,th=NULL) {
  sel <- as.numeric(prob > th)
  truth <- rep(0, length(prob)); truth[true_idx] <- 1
  TP <- sum(sel==1 & truth==1)
  FP <- sum(sel==1 & truth==0)
  FN <- sum(sel==0 & truth==1)
  Precision <- ifelse((TP+FP)==0, 0, TP/(TP+FP))
  Recall <- ifelse((TP+FN)==0, 0, TP/(TP+FN))
  F1 <- ifelse((Precision+Recall)==0, 0, 2*Precision*Recall/(Precision+Recall))
  FDP <- ifelse((TP+FP)==0, 0, FP/(TP+FP))
  Hamming <- sum(abs(sel-truth))
  data.frame(FDP, Precision, Recall, F1, Hamming)
}

# ================================================================
# 独立方法（不使用 TVS 框架）：直接在完整数据上运行并返回选中变量 + 时间
run_bart <- function(X, Y, topq = 50, niter = 1000, ntree = 50) {
  start_time <- Sys.time()
  res <- tryCatch({
    fit <- BART::wbart(x.train = X, y.train = Y, ndpost = niter, ntree = ntree, printevery = 10000, sparse = FALSE)
    imp <- fit$varcount.mean
    if (is.null(imp)) imp <- colMeans(fit$varcount)
    # 取 topq 个变量作为被选择
    top_idx <- order(imp, decreasing = TRUE)[1:min(topq, length(imp))]
    list(selected_vars = sort(unique(top_idx)), time = as.numeric(difftime(Sys.time(), start_time, units = "secs")))
  }, error = function(e) {
    list(selected_vars = integer(0), time = as.numeric(difftime(Sys.time(), start_time, units = "secs")))
  })
  return(res)
}

run_dart <- function(X, Y, topq = 50, niter = 1000, ntree = 50) {
  start_time <- Sys.time()
  res <- tryCatch({
    fit <- BART::wbart(x.train = X, y.train = Y, ndpost = niter, ntree = ntree, sparse = TRUE, printevery = 10000)
    imp <- if (!is.null(fit$varcount.mean)) fit$varcount.mean else colMeans(fit$varcount)
    top_idx <- order(imp, decreasing = TRUE)[1:min(topq, length(imp))]
    list(selected_vars = sort(unique(top_idx)), time = as.numeric(difftime(Sys.time(), start_time, units = "secs")))
  }, error = function(e) {
    list(selected_vars = integer(0), time = as.numeric(difftime(Sys.time(), start_time, units = "secs")))
  })
  return(res)
}

# 修改 LASSO 函数 - 选择系数非零的变量
run_lasso <- function(X, Y, topq = 5) {
  start_time <- Sys.time()
  res <- tryCatch({
    cv_fit <- glmnet::cv.glmnet(X, Y, alpha = 1, family = "gaussian")
    lasso_fit <- glmnet::glmnet(X, Y, alpha = 1, lambda = cv_fit$lambda.min, family = "gaussian")
    beta <- as.vector(coef(lasso_fit))[-1]  # 获取系数（去掉截距）
    
    # 修改：选择所有系数非零的变量，不再限制topq
    sel_idx <- which(abs(beta) > 1e-8)  # 选择系数绝对值大于阈值（非零）的变量
    
    list(selected_vars = sort(unique(sel_idx)), 
         time = as.numeric(difftime(Sys.time(), start_time, units = "secs")))
  }, error = function(e) {
    list(selected_vars = integer(0), 
         time = as.numeric(difftime(Sys.time(), start_time, units = "secs")))
  })
  return(res)
}

# 修改 SSLASSO 函数 - 选择系数非零的变量
run_sslasso <- function(X, Y, topq = 5) {
  start_time <- Sys.time()
  res <- tryCatch({
    fit <- SSLASSO::SSLASSO(X, Y, lambda0 = seq(0.1, 100, length = 10), lambda1 = 0.1)
    beta <- fit$beta[, ncol(fit$beta)]  # 获取系数向量
    
    # 修改：选择所有系数非零的变量，不再限制topq
    sel_idx <- which(abs(beta) > 1e-8)  # 选择系数绝对值大于阈值（非零）的变量
    
    list(selected_vars = sort(unique(sel_idx)), 
         time = as.numeric(difftime(Sys.time(), start_time, units = "secs")))
  }, error = function(e) {
    list(selected_vars = integer(0), 
         time = as.numeric(difftime(Sys.time(), start_time, units = "secs")))
  })
  return(res)
}
run_sis <- function(X, Y, topq = 5) {
  start_time <- Sys.time()
  p <- ncol(X)
  
  # 每个变量与 Y 做独立回归，得到相关性强度
  corr <- apply(X, 2, function(x) abs(cor(x, Y)))
  
  # 取前 topq
  sel_idx <- order(corr, decreasing = TRUE)[1:topq]
  
  time_used <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  list(selected_vars = sort(sel_idx), time = time_used)
}


# ================================================================
# 6️⃣ compare_methods：九方法对比（包含 TVS 系列与 独立方法）
# ================================================================
compare_methods <- function(n_runs = 1, n = 3000, p = 200, topq = 10, 
                            niter = 200, ntree = 75, batchsize = 100) {
  metrics_names <- c("FDP", "Precision", "Recall", "F1", "Hamming", "Time")
  method_list <- c("BART", "DART", "LASSO", "SSLASSO",
                   "SIS")
  
  
  # 初始化存放矩阵
  res_mats <- lapply(method_list, function(x) matrix(NA, n_runs, length(metrics_names)))
  names(res_mats) <- method_list
  
  for (run in 1:n_runs) {
    cat("=== run", run, "===\n")
    set.seed(2023 + run)
    d <- generate_data(n, p); X <- d$X; Y <- d$Y; signal_vars <- d$signal
    
    fix_stop=150
    binary_reward=TRUE
    
    # 5. 独立 BART（不使用 TVS_stream_generic）
    cat("运行 独立 BART...\n")
    sb <- run_bart(X, Y, niter = niter, ntree = ntree)
    prob_bart <- rep(0, p); prob_bart[sb$selected_vars] <- 1
    ev_sb <- evaluate_selection(prob_bart, signal_vars,th=1e-3)
    res_mats[["BART"]][run, 1:5] <- unlist(ev_sb); res_mats[["BART"]][run, 6] <- sb$time
    
    set.seed(2023 + run)
    d <- generate_data(n, p); X <- d$X; Y <- d$Y; signal_vars <- d$signal
    
    
    # 6. 独立 DART
    cat("运行 独立 DART...\n")
    sdart <- run_dart(X, Y, niter = niter, ntree = ntree)
    prob_dart <- rep(0, p); prob_dart[sdart$selected_vars] <- 1
    ev_sdart <- evaluate_selection(prob_dart, signal_vars,th=1e-3)
    res_mats[["DART"]][run, 1:5] <- unlist(ev_sdart); res_mats[["DART"]][run, 6] <- sdart$time
    
    set.seed(2023 + run)
    d <- generate_data(n, p); X <- d$X; Y <- d$Y; signal_vars <- d$signal
    
    
    # 7. 独立 LASSO
    cat("运行 独立 LASSO...\n")
    slasso <- run_lasso(X, Y, topq = topq)
    prob_lasso <- rep(0, p); prob_lasso[slasso$selected_vars] <- 1
    ev_lasso <- evaluate_selection(prob_lasso, signal_vars,th=1e-3)
    res_mats[["LASSO"]][run, 1:5] <- unlist(ev_lasso); res_mats[["LASSO"]][run, 6] <- slasso$time
    
    set.seed(2023 + run)
    d <- generate_data(n, p); X <- d$X; Y <- d$Y; signal_vars <- d$signal
    
    
    # 8. 独立 SSLASSO
    cat("运行 独立 SSLASSO...\n")
    sss <- run_sslasso(X, Y, topq = topq)
    prob_sss <- rep(0, p); prob_sss[sss$selected_vars] <- 1
    ev_sss <- evaluate_selection(prob_sss, signal_vars,th=1e-3)
    res_mats[["SSLASSO"]][run, 1:5] <- unlist(ev_sss); res_mats[["SSLASSO"]][run, 6] <- sss$time
    
    set.seed(2023 + run)
    d <- generate_data(n, p); X <- d$X; Y <- d$Y; signal_vars <- d$signal
    
    # SIS
    cat("运行 SIS...\n")
    ssis <- run_sis(X, Y, topq = topq)
    prob_sis <- rep(0, p); prob_sis[ssis$selected_vars] <- 1
    ev_sis <- evaluate_selection(prob_sis, signal_vars,th=1e-3)
    res_mats[["SIS"]][run, 1:5] <- unlist(ev_sis)
    res_mats[["SIS"]][run, 6] <- ssis$time
    
  }
  
  return(res_mats)
}

# ================================================================
# 7️⃣ 绘图函数（6 指标柱状图）
# ================================================================
plot_combined_metrics <- function(results, save_path = "tvs_rfe_comparison_regression.pdf") {
  metrics <- c("FDP", "Precision", "Recall", "F1", "Hamming", "Time")
  method_names <- names(results)
  n_methods <- length(method_names)
  
  colors <- rainbow(n_methods)
  names(colors) <- method_names
  
  pdf(save_path, width = 15, height = 10)
  par(mfrow = c(2, 3), mar = c(8, 4, 3, 1), oma = c(2, 2, 3, 1))
  
  for (k in seq_along(metrics)) {
    m <- k  # 用数字索引代替字符串
    metric_name <- metrics[k]
    
    means <- sapply(method_names, function(method) {
      data <- results[[method]][, m]
      data_clean <- data[!is.na(data) & is.finite(data)]
      if (length(data_clean) == 0) return(0)
      mean(data_clean)
    })
    
    sds <- sapply(method_names, function(method) {
      data <- results[[method]][, m]
      data_clean <- data[!is.na(data) & is.finite(data)]
      if (length(data_clean) <= 1) return(0)
      sd(data_clean)
    })
    
    y_max <- max(means + sds, na.rm = TRUE)
    y_lim <- c(0, y_max * 1.25)
    
    bp <- barplot(means, main = metric_name, ylab = "Value",
                  col = colors, ylim = y_lim,
                  las = 2, cex.names = 0.7)
    
    arrows(bp, pmax(means - sds, 0), bp, means + sds,
           length = 0.05, angle = 90, code = 3)
  }
  
  mtext("TVS + Baselines Performance Comparison (Regression)",
        outer = TRUE, cex = 1.3, font = 2, line = 1)
  
  legend("bottom", legend = method_names, fill = colors, ncol = 3, bty = "n", inset = -0.08, xpd = TRUE)
  dev.off()
  message("✅ 图像已保存至: ", save_path)
}

# ================================================================
# 8️⃣ 运行对比实验（示例）
# ================================================================
cat("开始运行 TVS-RFE 对比实验（示例参数）...\n")

# 建议先用较小规模做测试
results <- compare_methods(n_runs = 50, n = 500, p = 100, topq = 5, niter = 300, ntree = 50, batchsize = 50)

# 生成并保存图
plot_combined_metrics(results, save_path = "tvs_offline_comparison_regression.pdf")

# 打印摘要
summary_method <- function(mat) {
  clean_mat <- mat
  clean_mat[!is.finite(clean_mat)] <- NA
  mu <- colMeans(clean_mat, na.rm = TRUE)
  sdv <- apply(clean_mat, 2, sd, na.rm = TRUE)
  out <- rbind(mean = mu, sd = sdv)
  return(round(out, 4))
}

method_names <- names(results)
for (method in method_names) {
  cat("\n=== Summary:", method, "===\n")
  print(summary_method(results[[method]]))
}
