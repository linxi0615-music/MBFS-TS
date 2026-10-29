# 离线
library(randomForest)
library(e1071)
library(caret)
library(nnet)
library(stats)
library(glmnet)
set.seed(2025)
# ============================================================
# 1. 三分类数据生成函数（适用于离线版本）
# ============================================================
generate_multiclass_data1 <- function(n = 300, p = 1000, n_signal = 5, sigma = 3) {
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  signal_vars <- 1:n_signal  # 前5个为真实信号特征
  
  # 线性组合
  linear_comb_0 <- 3.2 * X[, 1] + 2.8 * X[, 2] - 1.5 * X[, 3] +
    2.1 * X[, 4] - 1.8 * X[, 5] + rnorm(n, 0, sigma) + 2
  linear_comb_1 <- -2.5 * X[, 1] + 1.9 * X[, 2] + 2.3 * X[, 3] +
    -1.7 * X[, 4] + 2.0 * X[, 5] + rnorm(n, 0, sigma) + 0
  linear_comb_2 <- -1.8 * X[, 1] - 2.2 * X[, 2] + 1.2 * X[, 3] +
    -2.4 * X[, 4] - 1.5 * X[, 5] + rnorm(n, 0, sigma) - 2
  
  # 非线性项
  nonlinear_0 <- 0.8 * sin(pi * X[, 1] * X[, 2]) + 0.5 * (X[, 3]^2)
  nonlinear_1 <- 0.6 * cos(pi * X[, 3] * X[, 4]) + 0.4 * (X[, 5]^2)
  nonlinear_2 <- 0.7 * sin(pi * X[, 5] * X[, 1]) + 0.3 * (X[, 2]^2)
  
  # 合并分数
  score_0 <- linear_comb_0 + nonlinear_0
  score_1 <- linear_comb_1 + nonlinear_1
  score_2 <- linear_comb_2 + nonlinear_2
  
  # Softmax概率
  scores <- cbind(score_0, score_1, score_2)
  exp_scores <- exp(scores)
  probs <- exp_scores / rowSums(exp_scores)
  
  # 生成标签
  Y <- apply(probs, 1, function(p) sample(0:2, 1, prob = p))
  
  return(list(X = X, Y = Y, signal_vars = signal_vars))
}

generate_multiclass_data <- function(n = 300, p = 1000, n_signal = 5, sigma = 2) {
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  signal_vars <- 1:n_signal  # 前 n_signal 个为真实信号特征
  
  # 线性成分
  linear_pos <- 2.5 * X[, 1] + 2.0 * X[, 2] - 1.2 * X[, 3] +
    1.8 * X[, 4] - 1.5 * X[, 5] + rnorm(n, 0, sigma) + 1
  linear_neg <- -2.0 * X[, 1] - 1.5 * X[, 2] + 1.0 * X[, 3] -
    1.2 * X[, 4] + 1.3 * X[, 5] + rnorm(n, 0, sigma) - 1
  
  # 非线性成分（交互项 + 周期项）
  nonlinear_pos <- 0.6 * sin(pi * X[, 1] * X[, 2]) +
    0.4 * cos(2 * pi * X[, 3]) +
    0.5 * (X[, 4]^2 - X[, 5]^2)
  nonlinear_neg <- 0.5 * cos(pi * X[, 2] * X[, 3]) +
    0.3 * sin(2 * pi * X[, 4]) +
    0.4 * (X[, 1] * X[, 5])
  
  # 合并得分
  score_pos <- linear_pos + nonlinear_pos
  score_neg <- linear_neg + nonlinear_neg
  
  # Sigmoid 概率
  prob_pos <- 1 / (1 + exp(-(score_pos - score_neg)))
  
  # 生成二分类标签
  Y <- rbinom(n, 1, prob_pos)
  
  return(list(X = X, Y = Y, signal_vars = signal_vars))
}
# ============================================================
# 2. 离线奖励函数集合（修改Logistic奖励函数）
# ============================================================

# ============================================================
# 3. TVS离线特征选择算法集合
# ============================================================
# 3.1 TVS-RF离线版
TVS_RF_offline <- function(Y, X, topq = 10, fix_stop = 200, binary_reward = TRUE) {
  result <- TVS_generic(
    Y = Y, X = X,
    reward_func = reward_rf_offline,
    reward_args = list(ntree = 85),
    topq = topq, fix_stop = fix_stop,
    binary_reward = binary_reward,
    M_theta = 6, update_mode = "single"
  )
  
  prob <- result$A[, result$final_iter + 1] / 
    (result$A[, result$final_iter + 1] + result$B[, result$final_iter + 1])
  
  list(
    prob = prob, A = result$A, B = result$B,
    time = result$time, iterations = result$final_iter
  )
}

# ============================================================
# 4. TVS_generic框架（通用离线TVS框架）
# ============================================================
TVS_generic <- function(Y, X, reward_func, reward_args = list(),
                        topq = 20, fix_stop = 200, binary_reward = TRUE,
                        M_theta = 6, delta = 1e-3, q = NULL, update_mode = "batch") {
  if (is.null(q)) q <- delta
  start_time <- Sys.time()
  p <- ncol(X)
  A <- matrix(1, p, fix_stop + 1)
  B <- matrix(1, p, fix_stop + 1)
  phi_q <- stats::qnorm(1 - q); if (!is.finite(phi_q)) phi_q <- 3
  final_iter <- fix_stop
  
  # 假设真实的重要特征（这里需要根据你的实际情况修改）
  # 如果没有真实标签，这些指标将无法计算
  true_important <- NULL  # 你需要在这里指定真实的重要特征索引
  
  for (t in 1:fix_stop) {
    Aprev <- A[, t]; Bprev <- B[, t]
    Ni <- Aprev + Bprev; Ni[Ni <= 0] <- 1
    mu_hat <- Aprev / Ni; mu_hat[is.na(mu_hat)] <- 0
    L1t <- log(12 * (max(2, p)^2) * (t^2) / delta + 1)
    C_t <- L1t / (phi_q^2 + 1e-8)
    
    # draw M_theta theta samples
    theta_mat <- matrix(0, nrow = M_theta, ncol = p)
    for (k in 1:M_theta) {
      sd_vec <- sqrt(C_t / Ni); sd_vec[!is.finite(sd_vec)] <- sqrt(C_t)
      theta_k <- stats::rnorm(p, mean = mu_hat, sd = sd_vec)
      theta_k <- pmin(pmax(theta_k, 0), 1)
      theta_mat[k, ] <- theta_k
    }
    
    # Oracle on mu_hat
    if (!is.null(topq) && topq > 0 && topq < p) {
      S_hat <- order(mu_hat, decreasing = TRUE)[1:topq]
    } else {
      S_hat <- which(mu_hat > 0.5)
      if (length(S_hat) == 0) {
        S_hat <- which.max(mu_hat)
      }
    }
    
    # compute S_tilde and gaps
    S_tilde_list <- vector("list", M_theta); delta_tilde <- numeric(M_theta)
    for (k in 1:M_theta) {
      th <- theta_mat[k, ]
      if (!is.null(topq) && topq > 0 && topq < p) {
        S_tilde <- order(th, decreasing = TRUE)[1:topq]
      } else {
        S_tilde <- which(th > 0.5)
        if (length(S_tilde) == 0) S_tilde <- which.max(th)
      }
      S_tilde_list[[k]] <- S_tilde
      delta_tilde[k] <- sum(th[S_tilde]) - sum(th[S_hat])
    }
    
    # stop if all equal
    all_same <- all(sapply(S_tilde_list, function(s) length(s) == length(S_hat) && all(sort(s) == sort(S_hat))))
    if (all_same) { final_iter <- t; break }
    
    kstar <- which.max(delta_tilde)
    S_tilde_star <- S_tilde_list[[kstar]]
    exch_set <- sort(unique(c(setdiff(S_hat, S_tilde_star), setdiff(S_tilde_star, S_hat))))
    
    if (length(exch_set) == 0) {
      pull_idx <- which.min(Ni)
    } else {
      pull_idx <- exch_set[which.min(Ni[exch_set])]
    }
    
    # form gammas as S_hat indicators
    gammas <- integer(p); gammas[S_hat] <- 1L
    
    # call reward_func safely
    args <- c(list(Y = Y, X = X, gammas = gammas), reward_args)
    rfull <- tryCatch(do.call(reward_func, args), error = function(e) rep(0, p))
    
    if (!is.numeric(rfull) || length(rfull) != p) {
      if (is.numeric(rfull) && length(rfull) == sum(gammas)) {
        full <- rep(0, p); full[which(gammas == 1)] <- rfull
        rfull <- full
      } else {
        rfull <- rep(0, p)
      }
    }
    
    # normalize to [0,1]
    if (max(rfull, na.rm = TRUE) > 0) {
      rprob <- pmin(pmax(rfull / max(rfull, na.rm = TRUE), 0), 1)
    } else rprob <- rep(0, p)
    
    # update A/B - 按照你的要求修改
    Anew <- Aprev; Bnew <- Bprev
    
    # 第一步：根据 gammas == 1 更新A/B
    idxs <- which(gammas == 1)
    if (length(idxs) > 0) {
      vals <- stats::rbinom(length(idxs), 1, prob = rprob[idxs])
      Anew[idxs] <- Anew[idxs] + vals
      Bnew[idxs] <- Bnew[idxs] + (1 - vals)
    }
    
    # 第二步：对 pull_idx 执行额外的更新
    Anew[pull_idx] <- Anew[pull_idx] + 1
    
    A[, t + 1] <- Anew; B[, t + 1] <- Bnew
  } # end loop
  
  end_time <- Sys.time()
  
  # trim matrices to final_iter
  A_out <- A[, 1:(final_iter + 1), drop = FALSE]
  B_out <- B[, 1:(final_iter + 1), drop = FALSE]
  final_prob <- A_out[, ncol(A_out)] / (A_out[, ncol(A_out)] + B_out[, ncol(B_out)])
  
  
  final_selected <- which(final_prob > 0.5)
  
  
  return(list(
    A = A_out,
    B = B_out,
    prob = final_prob,
    final_iter = final_iter,
    time = as.numeric(difftime(end_time, start_time, units = "secs"))
  ))
}

# ============================================================
# 5. 独立方法特征选择实现
# ============================================================
# 5.1 独立RF特征选择
rf_feature_selection <- function(Y, X) {
  start <- Sys.time()
  Y <- as.factor(Y)
  rf <- randomForest(x = X, y = Y, ntree = 50, importance = TRUE)
  imp <- importance(rf, type = 2)
  imp <- as.vector(imp)
  max_imp <- max(imp, na.rm = TRUE)
  prob <- if (max_imp > 0) imp / max_imp else rep(0, length(imp))
  time <- as.numeric(difftime(Sys.time(), start, units = "secs"))
  list(prob = prob, time = time)
}

# 5.2 独立逻辑回归特征选择
logistic_feature_selection <- function(Y, X) {
  start <- Sys.time()
  Y <- as.factor(Y)
  df <- data.frame(Y = Y, X)
  fit <- cv.glmnet(as.matrix(X), as.factor(Y),
                   family = "multinomial", type.multinomial = "grouped",
                   alpha = 1, nfolds = 5, maxit = 1e5)
  cf  <- coef(fit, s = "lambda.1se")
  imp <- as.numeric(Reduce(`+`, lapply(cf, function(m) abs(as.matrix(m))))[-1, ])
  prob <- imp / max(imp)
  time <- as.numeric(difftime(Sys.time(), start, units = "secs"))
  list(prob = prob, time = time)
}

# 5.3 独立SVM特征选择
svm_feature_selection <- function(Y, X) {
  start <- Sys.time()
  Y <- as.factor(Y)
  svm_model <- svm(x = X, y = Y, kernel = "linear", cost = 1, scale = TRUE)
  if (!is.null(svm_model$coefs) && !is.null(svm_model$SV)) {
    w_mat <- t(svm_model$coefs) %*% svm_model$SV
    imp <- colSums(abs(w_mat))
  } else {
    imp <- rep(0, ncol(X))
  }
  max_imp <- max(imp, na.rm = TRUE)
  prob <- if (max_imp > 0) imp / max_imp else rep(0, length(imp))
  time <- as.numeric(difftime(Sys.time(), start, units = "secs"))
  list(prob = prob, time = time)
}


# ============================================================
# 6. 特征选择评估函数
# ============================================================
evaluate_selection <- function(prob, true_idx, threshold = 0.5) {
  sel <- as.numeric(prob > threshold)  # 基于阈值选择特征
  truth <- rep(0, length(prob)); truth[true_idx] <- 1  # 真实信号标记
  
  TP <- sum(sel == 1 & truth == 1)  # 真正例
  FP <- sum(sel == 1 & truth == 0)  # 假正例
  FN <- sum(sel == 0 & truth == 1)  # 假负例
  
  Precision <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)  # 精确率
  Recall <- ifelse((TP + FN) > 0, TP / (TP + FN), 0)     # 召回率
  F1 <- ifelse((Precision + Recall) > 0, 2 * Precision * Recall / (Precision + Recall), 0)  # F1分数
  FDP <- ifelse((TP + FP) > 0, FP / (TP + FP), 0)        # 假发现率
  Hamming <- sum(abs(sel - truth))                       # 汉明距离
  
  data.frame(
    FDP = FDP,
    Precision = Precision,
    Recall = Recall,
    F1 = F1,
    Hamming = Hamming,
    Time = NA  # 预留时间字段
  )
}

# ============================================================
# 固定随机种子
num <- 50  # 重复运行次数
all_results <- list()  # 存储所有次运行的结果

# 循环运行num次
for (run in 1:num) {
  cat("\n===== 第", run, "次运行 =====", "\n")
  set.seed(2023+run)
  
  # 生成离线数据集
  data <- generate_multiclass_data(n = 500, p = 100, n_signal = 5)
  X <- data$X
  Y <- data$Y
  signal_vars <- data$signal_vars  # 真实信号特征：1-5
  colnames(X) <- paste0("X", 1:ncol(X))  # 添加列名
  
  # 存储本次运行的结果（8种方法）
  results <- list()
  
  # ---------------------- 独立方法（4种） ----------------------
  cat("运行独立RF...\n")
  rf <- rf_feature_selection(Y, X)
  eval_ind_rf <- evaluate_selection(rf$prob, signal_vars, threshold = 0.5)
  eval_ind_rf$Time <- rf$time
  results[["RF"]] <- eval_ind_rf
  
  cat("运行独立Logistic...\n")
  logistic <- logistic_feature_selection(Y, X)
  eval_ind_logistic <- evaluate_selection(logistic$prob, signal_vars, threshold = 1e-3)
  eval_ind_logistic$Time <- logistic$time
  results[["Logistic"]] <- eval_ind_logistic

  
  
  # 保存本次运行结果
  all_results[[run]] <- results
}

# ============================================================
# 9. 结果汇总与均值计算（包含均值和标准差）
# ============================================================
# 提取所有方法名称（8种）
method_names <- names(all_results[[1]])

# 初始化结果存储
metrics <- c("FDP", "Precision", "Recall", "F1", "Hamming", "Time")
results_summary <- list()

# 为每个方法计算均值和标准差
for (method in method_names) {
  # 提取所有次运行的该方法结果
  method_results <- lapply(all_results, function(run_res) run_res[[method]])
  method_df <- do.call(rbind, method_results)
  
  # 计算均值和标准差
  means <- colMeans(method_df, na.rm = TRUE)
  sds <- apply(method_df, 2, sd, na.rm = TRUE)
  
  results_summary[[method]] <- list(
    mean = means,
    sd = sds
  )
}

# 创建详细的均值结果数据框（包含均值和标准差）
mean_results_detailed <- data.frame(
  Method = method_names
)

# 为每个指标添加均值和标准差列
for (metric in metrics) {
  mean_col <- paste0(metric, "_mean")
  sd_col <- paste0(metric, "_sd")
  
  mean_results_detailed[[mean_col]] <- sapply(method_names, function(m) results_summary[[m]]$mean[metric])
  mean_results_detailed[[sd_col]] <- sapply(method_names, function(m) results_summary[[m]]$sd[metric])
}

# 创建简化的均值结果数据框（用于绘图）
mean_results <- data.frame(
  Method = method_names
)

for (metric in metrics) {
  mean_results[[metric]] <- sapply(method_names, function(m) results_summary[[m]]$mean[metric])
}

# ============================================================
# 10. 结果可视化（显示均值和误差线）
# ============================================================
# 打印详细的均值结果（包含均值和标准差）
cat("\n===== 多次运行详细结果（均值±标准差） =====\n")
for (method in method_names) {
  cat("\n---", method, "---\n")
  for (metric in metrics) {
    mean_val <- round(results_summary[[method]]$mean[metric], 4)
    sd_val <- round(results_summary[[method]]$sd[metric], 4)
    cat(metric, ": ", mean_val, " ± ", sd_val, "\n")
  }
}
