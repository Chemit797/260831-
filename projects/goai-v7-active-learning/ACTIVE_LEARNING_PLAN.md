# 主动学习计划：从已验证框架到 V7 接口

## 现有框架

GOAI-AL v2.2 使用 Direct MLP 与冻结的 chemical/strain semantic assets。它已有正式五 seed 和几何分析，但这些结果只证明该框架下的实验协议/分析可以被审计；它们不是 V7 acquisition 实验。

## 下一阶段的最小研究问题

> 在固定初始标注集、候选池、预算、fold 和评估协议时，来自 V7 的某种 acquisition signal 是否优于 random/coverage/现有语义策略？

## 必须先冻结的设计

| 维度 | 要决定的内容 |
|---|---|
| 单位 | 一个 observation、perturbation、strain、chemical 或组合？ |
| 可见性 | 每轮哪些标签可见，哪些严格 hold-out？ |
| 预测接口 | V7 返回 prediction、ensemble dispersion、MC estimate，还是 representation？ |
| acquisition | random、coverage、uncertainty、hybrid 的明确定义与 tie-breaker |
| 预算 | 初始集、每轮 batch、总轮数和停止条件 |
| 评估 | 逐蛋白 R² median 主指标；RMSE/MAE/global R²/PCC/coverage guardrail |
| 防泄漏 | 每轮独立 fit 的标准化/特征步骤；绝不看隐藏 test proteome |

## 建议实验顺序

1. **接口 smoke test**：只使用一个 seed、一小轮数，证明 V7 输入/输出、候选池和日志可串联。
2. **random 对照**：相同预算下 random acquisition；验证基础循环没有错误。
3. **单一 V7 signal**：只替换 acquisition signal，不改模型、数据、预算。
4. **多 seed / slice**：比较跨 seed、strain/chemical/entity slice 与 bootstrap 结果。
5. **消融**：zero/shuffle 与 plate/instrument 相关 guardrail，排除错误的信息泄漏或 proxy。

任何一步不通过，应写成失败模式，而不是直接进入更大搜索。
