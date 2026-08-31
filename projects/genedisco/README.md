# GeneDisco / DiscoBAX：历史主动学习基线

## 它为什么重要

GeneDisco 是主动学习入门阶段的基线复现。它提供了一个独立于当前 GO-AI 数据集的参照：如何固定任务、选择 acquisition、重复 seed、记录评估，以及区分可下载数据和真正需要保存的研究结论。

## 保留什么

* DiscoBAX 源码、README、`REPRODUCTION.md`、论文 PDF；
* 24-job IL-2 三 seed pilot 的配置、结果、日志和指标；
* Zenodo dataset `10202590` 的版本/哈希/下载说明。

## 不保留什么

`data/` 约 640 MiB 的公开缓存/ZIP 不进入默认迁移。它不是唯一证据，且复现说明已经固定来源与哈希。

## 如何阅读

先读 `REPRODUCTION.md`，再读 pilot 的结果摘要。不要将 GeneDisco 的指标直接与 GO-AI/V7 的逐蛋白 R² 比较；它的价值是实验方法和主动学习纪律，而非同一 benchmark 上的分数。
