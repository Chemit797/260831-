# GO-AI 第一阶段与 M12 档案

这是熟悉扰动预测任务、调模型和形成 M12 交付的历史阶段，不是当前 V7-AL 主线。

## 保留包

* `GOAI_离线实验与M12资料包_20260816`：权威轻量阅读包，含发布源码、模型台账、历史文档和关键回执；
* 16 个 M12 权重：M2、M6、M9 及 OP3 encoder，逐个与 `weights/manifest.json` 校验；
* 最终 submission zip：`GOAI-M12.0_submission_candidate_20260815.zip`；
* 关闭支线的源码/结果说明：M10 transfer、RNA transfer、batch audit。

## 不应误读的内容

M10.0–M10.8 在历史记录中均未晋级；其 7.7 GB models/OOF 不迁移。M11-vs-M12 原始 OOF 是可选的审计资料，不是新机器运行所需输入。

## 对当前工作的贡献

第一阶段解释了任务语言、数据边界和哪些旧探索被淘汰。它不应在未重新验证的情况下决定 V7 或主动学习的模型选择。
