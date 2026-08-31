# 当前状态（迁移日）

## 一句话

当前研究问题是：如何在 GO-AI 扰动预测数据上构建 **可审计、无泄漏、能够真正帮助采样决策** 的主动学习方案；成熟 V7 是最重要的性能/可复现基线。

## 已完成

| 线 | 已完成事实 | 证据位置 |
|---|---|---|
| V7 | `proteome_biostate_readout_v7_reproduction` 的 seed 42/43/44 三组完整复现 | artifact `goai-v7-three-seed-results-20260831` |
| V7 比较 | V7 可与 basic descriptor 五模型比较；保留的是预测/指标而非所有 comparator checkpoint | artifact `goai-v7-basic-descriptor-comparator-20260831` |
| AL | Direct MLP + frozen semantic features 的 GOAI-AL v2.2 正式五 seed、key geometry v4、transfer geometry v1 已完成 | `goai-al-*` artifacts |
| BCR | 结论已收束：内部记录无相对 flat MLP 的优势，released 比较中 V7 四场景胜出 | `goai-bcr-vs-v7-conclusion-evidence-20260831` |

## 尚未完成：最重要的接口缺口

GOAI-AL v2.2 当前使用 Direct MLP 和固定 semantic features；它没有读取 V7 checkpoint、V7 posterior、V7 uncertainty，也没有把 V7 的训练/验证协议作为 acquisition 的统一接口。下一阶段不是“再堆更多输出”，而是先写清：

1. V7 可暴露哪些预测/不确定性/表征；
2. acquisition unit 是何种 perturbation、strain、chemical 还是组合；
3. 每轮可观察的标签边界和候选池如何定义；
4. 与 random、coverage、uncertainty 等基线如何做同预算对比；
5. 如何保证 feature fitting、标准化、target 处理都只在该轮可见训练数据上完成。

## 当前优先级

1. 先恢复 V7 数据合同与三 seed 基线，使任何新机器可验证读取；
2. 把 AL 的 v2.2 现有正式结果、假设和限制整理为可读基线；
3. 设计 V7-to-AL 接口的最小 smoke experiment，而非直接大规模搜索；
4. 使用受控的 random/zero/shuffle/plate/instrument guardrail 判断改善是否可信。
