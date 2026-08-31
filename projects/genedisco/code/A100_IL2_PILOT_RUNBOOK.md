# DiscoBAX A100 IL-2 Pilot 复现执行规格

> 本文档是交给远程服务器 Codex 的完整任务说明。请先审计、实现和测试运行控制，再启动实验。不要跳过正确性检查，也不要将 smoke、策略重放、部分运行或论文数字标记为本机 FULL RUN。

## 1. 最终目标

在单张 NVIDIA A100 上完成一个范围受控、可审计、可断点续跑的 DiscoBAX pilot：

- assay：`schmidt_2021_il2`
- feature set：`achilles`
- seeds：`1000, 2000, 3000`
- acquisition methods：
  - `random`
  - `topuncertain`
  - `coreset`
  - `badge`
  - `ucb`
  - `topk_bax`
  - `levelset_bax`
  - `discobax`
- 每项正式实验：25 acquisition cycles，batch size 32
- 正式任务总数：`1 assay × 3 seeds × 8 methods = 24 FULL RUNS`
- 单张 A100 串行执行，一次只运行一个正式 job
- 估计总耗时：约 14-25 小时；以第一个完整 BAX job 的实测时间更新 ETA

这个 pilot 用来回答：在 IL-2 assay、三个指定 seed 下，DiscoBAX 是否呈现论文所报告的相对趋势。它不能单独支持“DiscoBAX 在所有 assay 上总体显著优于所有方法”的结论。

## 2. 远程目录约定

项目必须完整位于：

```text
/home/chenyuming/Project/active-learning/disco/DiscoBAX/
```

期望结构：

```text
DiscoBAX/
├── discobax/                         # Python package
├── reproduction/                     # configs, runner, audit, summarizer
├── tests/
├── data/
│   ├── DiscoBAX_GeneDisco_datasets.zip
│   └── release/data/                 # 解压后的精确官方 cache
├── results/                          # 运行时创建
├── logs/                             # 运行时创建
├── run_reproduction.sh
└── A100_IL2_PILOT_RUNBOOK.md
```

所有输出必须保存在这个 `DiscoBAX/` 大目录内部。不要把本项目的数据、结果或日志写到上一级 `disco/`；上一级未来还会并列放置 GeneDisco。

## 3. 来源与不可变合同

- 论文：ICML 2023，`DiscoBAX: Discovery of Optimal Intervention Sets in Genomic Experiment Design`
- 官方仓库：`https://github.com/amehrjou/DiscoBAX`
- 冻结的官方 commit：`84f01283bc7f6ab5f66b5ea2a63632b401cc0402`
- 官方数据：Zenodo record `10202590`
- 数据压缩包大小：`246500391` bytes
- 数据压缩包 MD5：`9f2fb895e32c85377e4cf1b2d2658ed9`
- 本项目是基于该 commit 的可审计 GPU port，不允许悄悄替换算法、模型结构、指标或论文参数

上传到服务器的源码压缩包有意排除了本机 `.git` worktree 指针，因为其中记录的是不可用于 Linux 的 Windows 路径。因此服务器目录可能不是 Git checkout。环境审计不得因为缺少 `.git` 直接崩溃；应记录本文档声明的 upstream commit，并对关键源码/config 生成 SHA-256 清单。若需要 Git 历史，应在独立目录重新 clone 官方 commit 后做 diff，不要伪造服务器源码的 Git 状态。

服务器 Codex 必须在 manifest 中记录：源码状态、环境版本、GPU 型号、数据路径、数据校验值、完整命令、开始/结束时间、退出码和运行时长。

## 4. 数据检查

在项目根目录执行：

```bash
cd /home/chenyuming/Project/active-learning/disco/DiscoBAX

md5sum data/DiscoBAX_GeneDisco_datasets.zip
stat -c '%s' data/DiscoBAX_GeneDisco_datasets.zip
```

只有 MD5 和大小都匹配时才允许继续。解压后的 cache 必须是：

```text
/home/chenyuming/Project/active-learning/disco/DiscoBAX/data/release/data
```

至少验证：

```bash
test -f data/release/data/achilles.h5
test -f data/release/data/schmidt_2021_il2.h5
test -f data/release/data/clusters_schmidt_2021_il2_achilles_0.01_topk_20_clusters_to_items.pkl
test -f data/release/data/clusters_schmidt_2021_il2_achilles_0.01_topk_items_to_20_clusters.pkl
```

禁止自动回退到其他 GeneDisco cache。项目历史中的旧 cache 已被证明与 Zenodo release 在数据规模和值上不等价。

## 5. 环境审计

先只做检查，不要盲目升级系统 CUDA、驱动或整个 Conda 环境：

```bash
nvidia-smi
which conda
conda env list
df -h /home/chenyuming/Project/active-learning/disco/DiscoBAX
```

本地已验证的参考栈：

```text
Python       3.8.20
PyTorch      2.4.1+cu124
CUDA runtime 12.4
GPyTorch     1.11
GeneDisco    1.0.5
SlingPy      0.2.11
NumPy        1.24.4
scikit-learn 1.3.2
```

服务器可使用与 A100 驱动兼容的 PyTorch CUDA build，但所有版本必须被记录。优先创建或复用 `genedisco-repro` 环境。安装本项目时使用 editable install，避免从 PyPI 安装另一个同名 DiscoBAX 覆盖当前源码：

```bash
conda run -n genedisco-repro python -m pip install -r reproduction/requirements-gpu.txt
conda run -n genedisco-repro python -m pip install --no-deps -e .
```

运行现有审计与测试：

```bash
conda run -n genedisco-repro python reproduction/audit_environment.py \
  --cache data/release/data

conda run -n genedisco-repro python -m pytest -q tests
```

要求至少保留 20 GB 可用磁盘空间。若空间不足，先报告，不要删除任何既有结果。

## 6. Pilot 的论文参数

创建独立配置：

```text
reproduction/configs/pilot_il2_3seed.json
```

内容必须等价于：

```json
{
  "name": "pilot_il2_3seed",
  "datasets": ["schmidt_2021_il2"],
  "methods": [
    "random",
    "topuncertain",
    "coreset",
    "badge",
    "ucb",
    "topk_bax",
    "levelset_bax",
    "discobax"
  ],
  "seeds": [1000, 2000, 3000],
  "common": {
    "feature_set_name": "achilles",
    "model_name": "bayesian_mlp",
    "acquisition_batch_size": 32,
    "num_active_learning_cycles": 25,
    "topk_percent": 0.01,
    "num_topk_clusters": 20,
    "bax_topk_kvalue": 2,
    "bax_level_set_c": 1.5,
    "bax_subset_select_subset_size": 10,
    "bax_noise_type": "additive",
    "bax_noise_lengthscale": 1.0,
    "bax_noise_outputscale": 1.0,
    "bax_num_samples_EIG": 20,
    "bax_num_samples_entropy": 20,
    "bax_entropy_average_mode": "arithmetic",
    "bax_batch_selection_mode": "topk_EIG",
    "bax_subset_select_num_samples": 20,
    "device": "cuda"
  }
}
```

不要使用官方应用的默认 BAX 参数。官方默认是 `K=5 / level=1.0 / S=20`，但论文筛选并用于表格的参数是 `K=2 / level=1.5 / S=10`。

Bayesian MLP 应保持：

- 输入 808 维
- hidden size 8
- `808 -> 8 -> 1`
- consistent MC dropout，`p=0.5`
- MSE loss
- batch size 64
- 最多 100 epochs
- Adam，learning rate `1e-3`
- weight decay `1e-4`
- early stopping patience 13

## 7. 八种方法的对照意义

| 方法 | 代表角度 |
|---|---|
| Random | 无策略基线 |
| Top Uncertainty | 模型不确定性 |
| Coreset | 数据空间覆盖/多样性 |
| BADGE | 不确定性与多样性结合 |
| UCB | 高预测值与不确定性 |
| Top-K BAX | 高值集合发现 |
| Levelset BAX | 阈值集合发现 |
| DiscoBAX | 高值与机制多样性结合 |

IL-2 是论文中 DiscoBAX 优势最清楚的 assay。选择它会形成有信息量的正向 pilot，但必须在报告中明确这是预先依据论文结果选择的 assay，不能将其包装成无偏的全数据集结论。

论文 IL-2 表格只用于完成后对照趋势，不得复制为本次运行结果：

| 方法 | Recall | Diversity | Overall |
|---|---:|---:|---:|
| Random | 5.2% | 32.8% | 13.1% |
| UCB | 13.1% | 67.3% | 29.6% |
| Top-K BAX | 13.6% | 69.8% | 30.8% |
| Levelset BAX | 14.2% | 70.0% | 31.5% |
| DiscoBAX | 15.7% | 75.0% | 34.3% |

## 8. 正式运行前必须处理的正确性风险

以下问题已经在官方代码和当前依赖组合中被发现。服务器 Codex 必须逐项复核，写测试并记录处理决定。不能仅因为代码能跑就忽略。

### 8.1 BAX baseline model 没有真正恢复

官方 `bax_sampling.py` 中多处调用：

```python
model.load_folder(temp_folder_name)
```

但 `load_folder` 返回新模型，返回值被忽略。这样每个 EIG 条件拟合可能从上一次被修改的模型继续，而不是从同一基础 posterior 开始。

应按 API 语义恢复并验证，例如赋回 `model`，并新增测试证明每个条件分支从相同 baseline state 开始。修复必须记录为“实现正确性修复”，不能伪称未修改官方行为。

### 8.2 Consistent MC dropout 的 train/eval 状态

已加载模型可能处于 `training=True`。该项目的 consistent dropout 在 train mode 下会为输入创建独立 mask，而官方注释声称候选点应共享同一个函数样本。

必须检查 `BayesianModule`、`ConsistentMCDropout` 和模型加载后的状态，新增测试验证：

- 一次函数样本内所有候选点使用一致的 dropout function sample；
- 不同 MC samples 仍然产生不同函数样本；
- 改变推理 batch size 不应改变候选点对应的函数样本语义。

不要仅添加一个未经验证的 `eval()` 就认为问题解决。

### 8.3 GPyTorch kernel 参数没有实际设为 1

在 GPyTorch 1.11 中：

```python
RBFKernel(lengthscale=1.0)
ScaleKernel(..., outputscale=1.0)
```

构造参数可能被忽略，实际 constrained value 约为 `0.693147`。必须显式赋值并测试：

```python
kernel.base_kernel.lengthscale = requested_lengthscale
kernel.outputscale = requested_outputscale
```

测试必须断言实际值为配置值，而不是只检查构造函数参数。

### 8.4 GPU 与官方 CPU MCD 路径可能改变随机序列

当前 GPU port 把 MC-dropout inference 放到 CUDA；官方训练完成后通常把模型移回 CPU。A100 上的精确 dropout RNG 和选择轨迹不保证与作者硬件或 CPU bitwise 一致。

要求保持算法、模型和采样次数一致，并记录硬件/软件栈；不要承诺逐点完全一致。目标是统计趋势复现，不是跨硬件 bitwise reproduction。

### 8.5 官方循环累计选择 832 个点

官方代码先随机选择初始 32 个点，然后执行 25 个循环，每轮再选择 32 个点，因此最终累计为：

```text
32 + 25 × 32 = 832
```

虽然论文文字容易被理解为总计 800，但论文表格来自官方执行路径。本 pilot 暂时保持官方循环语义，不要擅自改成 800。必须在最终报告中披露。

## 9. Smoke 与正式运行门槛

在启动 24 项正式矩阵前，必须依次通过：

1. 数据 MD5、大小和必需文件检查。
2. A100 可见，`torch.cuda.is_available()` 为 true，device name 正确。
3. 全部单元测试通过。
4. 新增的 baseline restore、dropout 语义、kernel 参数测试通过。
5. 执行一个 IL-2 DiscoBAX one-cycle smoke：完整 MC 参数，不得降低 `20/20/20`，只把 cycles 改为 1。
6. smoke 返回码为 0，生成 cycle 产物，日志无 OOM、NaN、设备不一致或 silent fallback to CPU。
7. 记录 smoke runtime 和 peak GPU memory。

Smoke 只能标记为 `SMOKE`，不能进入正式三 seed 均值。

## 10. 后台控制器要求

服务器 Codex 应扩展现有 `run_reproduction.sh`，加入独立 scope：

```bash
bash run_reproduction.sh start pilot
bash run_reproduction.sh status
bash run_reproduction.sh follow
bash run_reproduction.sh stop
bash run_reproduction.sh resume
bash run_reproduction.sh summarize
bash run_reproduction.sh doctor
```

也可以新增一个单入口 `run_il2_pilot.sh`，但不要让用户组合多条底层 Python 命令才能控制实验。

控制器必须满足：

- `start pilot` 默认启动上述 24 项，不启动 1500 项全论文矩阵。
- 使用 `nohup` 加独立 session/process group，使 SSH 断开不终止任务。
- 启动 runner 和 GPU monitor 时显式设置 `CUDA_VISIBLE_DEVICES=0`，本 pilot 只使用一张 A100；不得因为服务器暴露了 GPU 1 就自动占用第二张卡。
- PID/lock 防止重复启动两个 runner。
- `status` 显示：总任务、完成、失败、当前 method/seed、当前 cycle、运行时长和 ETA。
- `follow` 每 30 秒刷新；用户 Ctrl-C 只退出查看，不停止后台任务。
- `stop` 只终止经过校验的 runner process group，不误杀其他 Python/CUDA 任务。
- `resume` 按正式产物检查，跳过已完整成功的 job。
- 单张 A100 默认 concurrency=1；不要在未基准测试时并行多个 5.6 GB BAX job。
- 每个 job 独立日志，controller 另有总日志。
- 每个完成 job 立即原子更新 state/manifest，不等全部结束才写。
- 使用 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 和项目已有 deterministic 设置。
- 捕获 SIGTERM/SIGINT，停止当前 child 并把状态写成 interrupted，而不是伪装 completed。

## 11. 产物与断点续跑设计

正式输出使用独立目录，避免与本地旧的 `paper_full`、smoke 或 replay 混合：

```text
results/pilot_il2_3seed/
logs/pilot_il2_3seed/
```

推荐每个 job 使用独立子目录和独立 performance CSV：

```text
results/pilot_il2_3seed/jobs/<run_name>/
├── args.json
├── performance.csv
├── manifest.json
├── cycle_0/
├── ...
├── cycle_24/
└── run_results.pickle
```

不要让多个进程并发追加同一个 CSV。即便当前 concurrency=1，也应避免因中断重跑产生重复 final-cycle 行。

一个 job 只有同时满足以下条件才是 `FULL RUN / COMPLETE`：

- 子进程退出码为 0；
- 顶层 `run_results.pickle` 存在；
- 正好有 25 个成功 cycle 产物；
- final performance row 的 acquisition cycle 为 25；
- method、dataset、seed 和配置参数与 manifest 相符；
- 日志没有 traceback、OOM、NaN 或被杀记录。

如果 job 在 cycle 24 中断，它不是完整结果。现有应用不支持 cycle 内真正 checkpoint resume 时，应保留旧 attempt 供审计，并从该 job 起点重跑；不要把两次 attempt 的 CSV 行直接拼接成一个完整运行。

## 12. Random 的处理

本 pilot 只有三个 Random jobs，允许并建议走完整官方 25-cycle 路径，以保持 24 项全部具有统一 FULL RUN 产物。

Random acquisition 本身不使用模型；官方循环仍会每轮训练模型，因此这些训练在算法上是冗余的。若服务器 Codex决定加入快速 Random policy replay，必须：

- 保存正式脚本、配置、日志和逐 seed 选择索引；
- 用至少一个同 seed FULL RUN 验证每轮选点完全一致；
- 明确标记为 `POLICY REPLAY`，不得标记为 FULL RUN；
- 不得与 FULL RUN 数量混报。

为了避免 provenance 争议，本次三个 seed 默认全部完整运行，额外耗时相对整个 pilot 很小。

## 13. 推荐运行顺序

正式矩阵顺序建议：

1. `random` × 3
2. `topuncertain` × 3
3. `coreset` × 3
4. `badge` × 3
5. `ucb` × 3
6. `topk_bax` × 3
7. `levelset_bax` × 3
8. `discobax` × 3

这样可先完成便宜基线，再进入昂贵 BAX。第一个完整 `topk_bax` 和第一个完整 `discobax` 完成后分别更新各方法 ETA。不同方法耗时差异很大，不要用 Random 平均时间估算 BAX。

## 14. 监控内容

`status` 至少输出：

```text
profile          pilot_il2_3seed
GPU              NVIDIA A100 ...
state            RUNNING|STOPPED|FAILED|COMPLETE
progress         7/24
current          schmidt_2021_il2 / coreset / seed=2000
current cycle    11/25
elapsed          04:18:32
completed        7
failed           0
remaining        17
ETA              基于同方法已完成 job 或分类估计
controller log   ...
current job log  ...
```

建议额外记录：

```bash
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv -l 30
```

GPU 监控写入独立日志，并在 runner 完成或停止时一并终止。不要将监控进程遗留在后台。

## 15. 汇总口径

只读取每个完整 job 的 final cycle。对每个 method 的三个 seeds：

1. 分别计算 recall 均值。
2. 分别计算 diversity 均值。
3. overall 使用论文口径：

```text
overall = sqrt(mean_recall × mean_diversity)
```

不要先对每个 seed 算 overall 再取平均。

论文绘图代码使用 population standard deviation：

```text
std = np.std(values, ddof=0)
SEM = std / sqrt(n)
```

由于 `n=3` 很小，最终报告必须同时显示三个 seed 原始值、mean、std 和 SEM，并明确这是描述性 pilot，不做强统计显著性结论。

输出至少包括：

- `pilot_il2_3seed_final_per_seed.csv`
- `pilot_il2_3seed_summary.csv`
- `pilot_il2_3seed_manifest.jsonl`
- 每个 method 的 25-cycle recall/diversity/overall 曲线数据
- 论文 IL-2 数字与本次三 seed 均值的并排对照表
- 完成/失败/跳过任务清单
- 环境审计 JSON

## 16. 结果解释边界

允许的结论形式：

```text
在 IL-2 assay 和 seeds 1000/2000/3000 的 pilot 中，DiscoBAX 的三 seed
平均 recall/diversity/overall 相对这些指定基线表现为……
```

不允许的结论形式：

```text
DiscoBAX 已被证明在 GeneDisco 上总体最强。
完整复现了论文全部结果。
结果具有统计显著性。
```

还必须披露：IL-2 是依据论文中最明显的正向结果预先选择的 assay，存在 assay selection bias；三 seed 结果只适合作为复现 pilot。

## 17. 禁止事项

- 禁止手工抄论文数字填入本次结果文件。
- 禁止把 dry-run、smoke、部分 cycle 或 replay 称为 FULL RUN。
- 禁止因任务慢而私自把 BAX MC 参数从 `20/20/20` 降低。
- 禁止私自改变 batch size、cycle 数、Top 1%、cluster 数或 BAX 超参数。
- 禁止 silent CPU fallback。
- 禁止自动使用错误 cache。
- 禁止发现错误后删除日志或覆盖失败证据。
- 禁止用单 seed 或三 seed pilot 宣称论文总体结论。
- 禁止在一张 A100 上未经基准测试同时启动多个 BAX job。
- 禁止在用户不知情时启动 1500-job `paper` scope。

## 18. 服务器 Codex 的执行顺序

服务器 Codex 接到本文档后应实际完成，而不是只给方案：

1. 阅读本文档和项目 `REPRODUCTION.md`。
2. 审计目录、数据、环境、GPU、磁盘和当前进程。
3. 检查工作区已有修改，保护用户文件。
4. 审计并修正第 8 节的三个代码正确性问题。
5. 为修复新增聚焦测试，并运行完整测试集。
6. 创建 `pilot_il2_3seed.json`。
7. 扩展单入口后台控制器，加入 `pilot` scope 和进度显示。
8. dry-run，确认恰好 24 个正式任务，参数逐项正确。
9. 运行完整 MC 的一轮 DiscoBAX smoke。
10. 检查 smoke 产物、日志、GPU 使用和峰值显存。
11. 启动后台 pilot，并确认 SSH 断开安全。
12. 向用户给出 `status/follow/stop/resume/summarize` 的准确命令。
13. 运行过程中持续写 manifest；失败时继续其他 job，但明确报告。
14. 全部结束后自动汇总，并逐项验证 24 个 FULL RUN。

## 19. 启动前验收清单

只有以下项目全部为真才可正式启动：

- [ ] 当前路径为 `/home/chenyuming/Project/active-learning/disco/DiscoBAX`
- [ ] Zenodo ZIP 大小和 MD5 正确
- [ ] IL-2 HDF5、Achilles HDF5 和两个 cluster mapping 存在
- [ ] A100 被 PyTorch CUDA 正确识别
- [ ] 环境版本已记录
- [ ] 磁盘空间足够
- [ ] 已知 BAX 正确性问题已复核、修复并测试
- [ ] 全部测试通过
- [ ] pilot config 是 1 dataset、8 methods、3 seeds
- [ ] dry-run 正好打印 24 jobs
- [ ] 参数是论文值 `K=2 / level=1.5 / S=10 / MC=20/20/20`
- [ ] 完整 MC one-cycle smoke 成功
- [ ] controller lock/PID、断线存活、status、stop、resume 已测试
- [ ] 正式结果目录与旧结果隔离
- [ ] 用户知道本次只启动 pilot，不是 1500 项全矩阵

## 20. 完成标准

本任务只有满足以下条件才算完成：

- 24 个预定正式任务全部有明确状态；
- 所有成功任务都通过 FULL RUN 完整性验证；
- 失败任务保留日志并被明确列出；
- 汇总严格使用三个指定 seed 和论文口径；
- 原始逐 seed 数字、均值、SEM、学习曲线和环境信息齐全；
- 结果与论文并排比较，但没有伪造、替换或混淆来源；
- 给出基于真实运行结果的有限结论和剩余风险。
