# MBFS

**M**odel-**B**ased **F**eature **S**election with **T**hompson **S**ampling.

[![CI](https://github.com/<you>/mbfs/actions/workflows/ci.yml/badge.svg)](https://github.com/<you>/mbfs/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-%E2%89%A54.1-blue.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

将特征选择建模为马尔可夫决策过程（MDP），用 TD 学习估计状态价值，用 Beta 后验驱动的
Thompson Sampling 决定"下一个加入哪个特征"。最终每个特征得到一个后验均值 `theta`，
`theta > 0.5` 即被选中。支持分类与回归任务。

本仓库同时包含 **三个对比方法**（TVS / BART / SSLASSO）的实现，用于论文中的实验对比。

---

> [!WARNING]
> **运行环境不一致，请务必先读这一节。**
>
> | 方法 | 语言 | 是否能纯 Python 跑 | 依赖 |
> |------|------|-------------------|------|
> | **MBFS**（本库主算法） | **Python** | ✅ 是 | `numpy`, `pandas`, `scikit-learn`, `tqdm` |
> | **FSRL** | **Python** | ✅ 是 | `numpy`, `pandas`, `scikit-learn`, `tqdm` |
> | **TVS** | **R** | ✅ 是，需 R 环境，但是也可以改成 | 
> | **BART** | **R only** | ❌ 否，**仅支持 R 运行** | R 包 `BART`（或 `bartMachine`） |
> | **SSLASSO** | **R only** | ❌ 否，**仅支持 R 运行** | R 包 `sslasso` |
>
> - 只想用 MBFS → 安装 Python 依赖即可，无需 R。
> - 想复现全部对比结果 → 必须**同时**安装 Python 与 R。
> - **BART 和 SSLASSO 没有 Python 官方等价实现，也没有纳入本库的 Python 封装**，
>   其余选择R或者python均可。
>


## 安装


### R 环境（TVS / BART / SSLASSO）

> **这三个方法仅有 R 实现。



安装 R 包：

```r
# 在 R 交互式终端里执行
install.packages("BART")       # BART（也可用 bartMachine 替代，见下）
install.packages("SSLASSO")    # SSLASSO

# sslasso若无法安装则可从原论文找到链接安装：
```

验证安装：

```bash
Rscript -e 'library(BART);   cat("BART OK\n")'
Rscript -e 'library(sslasso);cat("sslasso OK\n")'
```

## Quick Start

### 只跑 MBFS（纯 Python，不需要 R）

```python
from sklearn.datasets import make_classification
from mbfs import MBFSThompson, evaluate_selection

X, y = make_classification(
    n_samples=400, n_features=200,
    n_informative=5, n_redundant=0, n_classes=2,
    random_state=0,
)

model = MBFSThompson(
    n_features=X.shape[1],          # 必须等于 X.shape[1]
    n_episodes=100,
    alpha_td=0.9,
    gamma=0.9,
    max_episode_steps=X.shape[1],
    early_stopping_patience=30,
    reward_estimator="logistic_l1",
    estimator_C=0.05,
    task="classification",
)

model.fit(X, y)

selected = model.get_selected_features()   # theta > 0.5 的特征索引
theta    = model.theta_                    # 每个特征的后验均值
metrics  = evaluate_selection(theta, true_idx=[0, 1, 2, 3, 4])
print(metrics)
```


## 算法原理

### MDP 建模

| 元素 | 定义 |
|------|------|
| 状态 `s` | 已选特征子集 |
| 动作 | 从未选特征中选一个加入 |
| 奖励 `r` | `perf(s ∪ {f}) − perf(s)`，即性能提升 |
| 转移 | 确定性：`s → s ∪ {f}` |
| 价值 `V(s)` | 从该子集出发的期望累积奖励 |

`perf(·)` 在分类任务下是 5 折 CV 准确率，在回归任务下是 5 折 CV 调整 R²。

### Episode 伪代码

```
for episode in 1..n_episodes:
    S = ∅
    for step in 1..max_episode_steps:
        # 1. Thompson Sampling 选特征
        f = argmax_i  Beta_sample(alpha_i, beta_i)   for i ∉ S

        # 2. 计算奖励
        r = perf(S ∪ {f}) − perf(S)

        # 3. TD 更新状态价值
        td_target = r + gamma * V(S ∪ {f})
        V(S) ← V(S) + alpha_td * (td_target − V(S))

        # 4. 用 TD 误差更新 Beta 后验
        if td_target > V(S):
            alpha_f += 1
        else:
            beta_f += 1

        S = S ∪ {f}

    theta = alpha / (alpha + beta)
```

### 为什么用 Thompson Sampling

每个特征维护一个 `Beta(α_f, β_f)` 后验。"该特征有助于提升性能"的信念越强，`α_f` 越大。
每步从所有未选特征的 Beta 分布采样并取最大者——**探索与利用自然平衡**：不确定性高的特征
采样值波动大，仍有机会被选中；已确认有效的特征采样值稳定偏高。


### `evaluate_selection`

```python
evaluate_selection(prob_scores, true_idx) -> dict
```

按 `prob_scores > 0.5` 选择特征，与 `true_idx` 对比。返回：

| Key | 含义 |
|-----|------|
| `FDP` | 错误发现率 `FP / (TP + FP)` |
| `Precision` | 精确率 |
| `Recall` | 召回率 |
| `F1` | F1 分数 |
| `Accuracy` | 特征级准确率 |
| `Hamming` | 汉明距离（误判特征数） |
| `selected_features` | 被选中的特征索引列表 |

> 阈值 0.5 固定在此函数中。若想用其他阈值，先用
> `model.get_selected_features(threshold=t)` 选特征，再自行计算指标。

---


### 为什么需要 R

MBFS 是 Python 实现，TVS / BART / SSLASSO 三个 baseline 是 R 实现。
论文里所有方法的对比表，需要**两种语言分别跑完后汇总**。本仓库提供的做法：

```
其余比如随机森林逻辑回归R或者python巨款
┌───────────────┐      ┌────────────────┐
│  Python 侧     │      │  R 侧           │
│  MBFS          │      │  TVS / BART     │
│  (mbfs/)       │      │  SSLASSO        │
└───────┬───────┘      └────────┬────────┘
        │                       │
        └─────► results/*.csv ◄─┘
                    │
                    ▼
          examples/aggregate_results.py
                    │
                    ▼
              论文对比表
```

## FSRL注意事项与局限
仅支持python，回归版本为原论文的拓展，提供原论文的FSRL库

---

## License

MIT
