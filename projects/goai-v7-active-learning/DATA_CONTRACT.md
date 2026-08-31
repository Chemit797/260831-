# V7 与主动学习数据合同

## 访问与敏感级别

官方 GO-AI metadata/proteome 是私有研究输入：只在 Drive 包中保存，GitHub 只保存逻辑 ID、SHA-256、大小和读取要求。不要上传原始 CSV 到 GitHub，也不要将隐藏 test proteome 作为普通输入。

| 逻辑数据 | 原始相对路径 | 用途 | 关键规则 |
|---|---|---|---|
| `goai-train-metadata` | `go-ai/WAYB_WAYC_metadata_train_val.csv` | 样本元数据、split/role 审计 | 按 `sample_ID` 显式对齐；split/role 不作普通 feature |
| `goai-train-proteome` | `go-ai/WAYB_WAYC_proteome_raw_train_val.csv` | 训练标签/观察矩阵 | `NaN` 是未观测，不是 0；使用共同有限值 mask |
| `v7-chemical-embeddings` | `go-ai-rebuild/data/chemical_embeddings.csv` | V7 化合物 512D block | 记录版本和实体映射 |
| `v7-strain-embeddings` | `go-ai-rebuild/data/strain_embeddings.csv` | V7 strain RAW4096 block | 同一训练 fold 内处理标准化/选择 |
| `al-chemberta-semantics` | `go-ai/data/processed/chemical_embeddings/chemberta_77m_mlm/` | AL 固定化学语义 | 不等同于 V7 化学 descriptor |
| `al-strain-semantics` | `go-ai/data/processed/entities/` | AL 固定菌株语义 | DHY210 语义缺失是明确限制 |

## 读写纪律

1. 任何 CSV 先按显式键/`sample_ID` 对齐，禁止依赖行顺序。
2. 训练、评估、特征选择、标准化与 target transformation 都只能在对应训练集合 fit。
3. `split_final`、`strain_role`、`chemical_role` 用于划分和审计，不可静默进入特征。
4. 外部 descriptor 必须写来源、版本、许可、实体匹配和获取日期。
5. 新下载包先验 SHA-256，再运行 schema/row-count/ID overlap 检查。
