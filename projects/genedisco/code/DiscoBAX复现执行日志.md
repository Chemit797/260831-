# DiscoBAX 复现执行日志

更新时间：2026-08-09

## 1. 当前状态

- 已完整阅读 21 页 ICML 2023 论文及附录，并核对算法、实验设置、表格和官方地址。
- 已克隆官方仓库，锁定提交 `84f01283bc7f6ab5f66b5ea2a63632b401cc0402`。
- `../official` 保持完全干净；GPU 改动位于 `../gpu` 的 `codex/gpu-port` 分支。
- 已下载并验证 Zenodo 记录 `10202590`，文件大小 `246500391` 字节，MD5 为 `9f2fb895e32c85377e4cf1b2d2658ed9`。
- 精确论文数据已解压到 `../data/release/data`，包含 5 个 assay、Achilles 和作者固定的 cluster 映射。
- 已复用之前的 `genedisco-repro` 环境并安装 `gpytorch==1.11`。
- 已完成 CPU/CUDA、噪声 sampler、梯度保留和配置预算共 10 项测试，全部通过。
- 已完成 exact-release GPU smoke、1 条 25-cycle Random 正式线，以及 1 条论文 MC 参数的 DiscoBAX 单-cycle 资源验证。
- 尚未运行完整的 1,500 个任务；完整矩阵需要较长 GPU 时间，建议分阶段运行或迁移到 A100 级机器。

## 2. 论文实验口径

正式配置位于 `reproduction/configs/paper_full.json`。

| 项目 | 论文口径 |
|---|---|
| assay | IFN-gamma、IL-2、Leukemia/NK、Tau、SARS-CoV-2 |
| feature | Achilles，808 维 |
| optimal interventions | phenotype Top 1% |
| diversity clustering | PCA 20 维，GMM 20 components，20 次初始化 |
| acquisition | 25 cycles，batch size 32 |
| seeds | 1000 到 20000，共 20 个 |
| model | Bayesian MLP + consistent MC dropout |
| DiscoBAX noise | additive Gaussian，RBF lengthscale 1 |
| DiscoBAX subset size | S=10 |
| Top-K BAX | K=2 |
| Levelset BAX | level=1.5 |
| methods | 论文表格中的 15 种方法 |

论文正文和官方 Slurm 脚本均使用 20 个 seeds；官方 README 中“重复 10 次”与这两处冲突，因此正式配置遵照论文和脚本的 20 次。

官方类默认 `bax_subset_select_subset_size=20`，但附录 D.3 的最终选择是 `S=10`，正式配置使用论文值 10。

## 3. 数据审计

不能把此前 GeneDisco 的工作区 cache 当作本论文精确输入。虽然 `achilles.csv` 的 SHA-256 完全相同，但由不同 HGNC 映射/预处理生成的 HDF5 已经发生语义差异。

| 数据 | release/旧 cache 行数 | 共同基因上的 mean absolute difference |
|---|---:|---:|
| Achilles | 17655 / 17651 | 0.35997923 |
| IFN-gamma | 18421 / 18412 | 0.22236371 |
| IL-2 | 18421 / 18412 | 0.21776505 |
| Leukemia/NK | 20147 / 20138 | 1.7631265 |
| Tau | 17984 / 17980 | 0.65836176 |
| SARS-CoV-2 | 19112 / 19112 | 1.0106699 |

因此：

- 论文复现只使用 `../data/release/data`；
- runner 默认指向 release，并在缺少 `achilles.h5` 时直接报错，不静默退回旧 cache；
- 旧 cache 的两条早期运行只视为兼容性验证，不进入论文汇总；
- `reproduction/compare_caches.py` 可重复上述语义审计。

## 4. GPU 改动边界

保持不变的内容：

- Bayesian MLP 结构、hidden size、loss、Adam、batch size 和 100 epochs；
- DiscoBAX/Top-K BAX/Levelset BAX 的目标和 EIG 公式；
- subset selection、noise model、指标和论文默认超参数；
- 官方结果目录的基本结构。

改动内容：

- 新增 `auto|cpu|cuda` 设备选择，替换噪声 sampler 中硬编码的 `.cuda()`；
- 让 MLP 的大批量 MC inference 真正在 GPU 执行，仅在 NumPy 接口处回 CPU；
- 对 Adversarial BIM 所需的输入梯度保留计算图，普通预测使用 `no_grad()`；
- 新增 `bax_subset_select_num_samples`，默认仍为官方值 20；只有 smoke 配置降低该值；
- performance CSV 新增 `device` 和 subset-select MC 预算字段；
- runner 增加 exact cache、日志、manifest、断点跳过和 cycle 数完整性检查。

GPU/CPU 浮点舍入可能造成极少量排序差异。代码已启用 PyTorch deterministic algorithms、关闭 cuDNN benchmark，并设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，但严格逐索引比较仍应固定同一软硬件栈。

## 5. 本机环境

```text
Python        3.8.20
PyTorch       2.4.1+cu124
CUDA runtime  12.4
GPU           NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB
GeneDisco     1.0.5, editable workspace install
slingpy       0.2.11
gpytorch      1.11
numpy         1.24.4
scikit-learn  1.3.2
```

论文官方 README 说明正式实验使用 A100 GPU。

## 6. 已完成结果

### 6.1 Exact-release DiscoBAX smoke

```text
dataset                    schmidt_2021_ifng
method                     discobax
seed                       1000
cycles                     1
subset size                2
EIG samples                1
entropy samples            2
subset-select MC samples   1
runtime                    17.1 s
status                     success
```

该配置只验证端到端设备和数据路径，不是论文性能结果。

### 6.2 Exact-release Random 正式线

除 seed 数只运行 1 个外，其余设置均为论文原值：

```text
dataset       schmidt_2021_ifng
method        random
seed          1000
cycles        25
batch size    32
runtime       275.9 s
Top-K recall  2.31%
diversity     10.00%
overall       4.81%
status        success
```

`overall = sqrt(Top-K recall * diversity)`，与论文表格定义一致。单个 seed 不应与论文 20-seed 均值直接比较。

### 6.3 论文 MC 参数 DiscoBAX 单-cycle

仅把 cycle 数从 25 缩为 1，其余 DiscoBAX 参数均使用论文正式值：

```text
dataset                    schmidt_2021_ifng
method                     discobax
seed                       1000
cycles                     1
subset size                10
EIG samples                20
entropy samples            20
subset-select MC samples   20
runtime                    482.1 s
observed peak GPU memory   about 5.6 GB
status                     success
```

该 cycle 的 Top-K recall、diversity 和 overall 均为 0，因为最初 64 个累计查询点没有命中 Top-1% target。这是资源/执行验证，不是方法性能结论。

## 7. 关键路径

```text
clean official source     ../official
GPU working tree          ../gpu
exact archive             ../data/DiscoBAX_GeneDisco_datasets.zip
exact cache               ../data/release/data
paper config              reproduction/configs/paper_full.json
exact results             ../results/exact_release
paper one-cycle result    ../results/paper_discobax_one_cycle
exact logs                ../logs/exact_release
tests                     tests
```

## 8. 继续运行

先扩展到 3 seeds、4 个核心方法：

```powershell
conda run -n genedisco-repro python reproduction/run_matrix.py `
  --config reproduction/configs/paper_full.json `
  --datasets schmidt_2021_ifng `
  --methods random,discobax,topk_bax,levelset_bax `
  --seeds 1000,2000,3000
```

全矩阵省略 filters，共 `5 x 15 x 20 = 1500` 个任务。runner 会跳过 cycle 数完整且已有 `run_results.pickle` 的任务。

