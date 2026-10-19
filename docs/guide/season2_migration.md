# Season2 公开题集迁移

主线 `tasks/` 使用 Season2 的 60 个算子，来自 dev 仓已合入的 PR 235，源提交为 `d5f55e7318cb72660187d6a5e4ca5ac92c022b58`。原 53 题及其原始 metadata 完整保存在 `archives/season1/tasks/`。

## 公开范围

每题提供 `desc.md`、`proto.yaml`、`golden.py`、`cases.yaml` 和 `cases.csv`，公开 case ID 为 1 到 20，共 1200 个公开用例。隐藏用例、内部筛选报告和内部工具不在本次同步范围内。

| 等级 | 名称 | 算子数 |
|---|---|---:|
| L1 | 基础 | 2 |
| L2 | 中级 | 13 |
| L3 | 高级 | 27 |
| L4 | 复杂 | 13 |
| L5 | 挑战级 | 5 |

算子名单与公开文件摘要见 `tasks/suite_manifest.json`。目录、proto 与 schema 入口保持一致。KV Cache 的公开接口要求显式返回两个输出 Tensor，允许直接返回被原地更新的输入缓冲别名，不要求额外复制；不引入省略返回值的接口扩展。

清单的 `files_sha256_policy` 为 `utf8_text_crlf_to_lf`：对公开 UTF-8 文本仅将 CRLF 换行转换为 LF 后计算 SHA256，不移除空格、BOM 或末尾换行。这样 Windows/Linux 检出不会造成摘要误报，实际文本变化仍会被检测。Season1 归档内容保持不变，其历史回归测试显式读取归档；历史提案与最终题集清单分开校验。

与 dev 题包相比，公开版统一了索引梯度算子的 schema、Golden 与文档入口；修正布尔值域的 YAML/CSV 类型差异。MoE 校验按 softmax 后的 top-k 语义处理并列与 NaN，已结束 token 的专家索引须使用 `num_expert` 哨兵，确定性的 `row_idx` 参与正常精度比较。

## 平台和性能数据

本次公开 910B2、910C、950PR 和 950DT 的 metadata，且只保留入选算子的公开 case ID。现有支持平台记录缺失基线时，以 `baseline_perf_us = 3 × t_hw_us` 补齐，含义为理论性能的三分之一，不是理论耗时的三分之一；估算值不伪装为实测值。Season2 禁止跨平台 metadata 回退，不借用 910B2 数据给其他平台评分。已有正数基线及理论值保持不变，仍需在新题集和目标软件环境下复核；本次题集切换不代表已完成性能验收。

910B2 的 1180 条公开记录中原缺失的 334 条基线、910C 的 1160 条公开记录中原缺失的 70 条基线，已按上述规则补齐。各文件 `_metadata.imputed_baseline_cases` 逐条列出估算记录，`imputed_baseline_formula` 和 `imputed_baseline_count` 记录公式与数量，便于后续以实测替换。950PR/950DT 当前记录均有基线，无需估算补值。Hash 算子当前文档仅声明支持 950，因此两个 910 metadata 文件均没有它的记录；不会为不支持项或整个平台缺失的记录凭空构造理论值。

`embedding_hash_table_lookup_or_insert` 和 `dynamic_mx_quant` 在当前 A3（910C）平台不支持，标记为 Unsupported / N/A；主线仍保留 60 个题包，A3 有效评测范围为 58 题、1160 个公开用例。这两题不应作为普通失败项扣分，也不应计入通过率和得分分母；排除名单已写入 `suite_manifest.json`，自动排除计分须由评测平台接入，本次不宣称该平台接入已完成。

950PR 和 950DT 各有 60 题、1200 条公开记录，来自用户提供的 `metadata_1015.zip`。公开文件采用 `baseline_perf_us = min(实测原值, max(10 × t_hw_us, 10μs))`；因此评分基线不等于原始耗时。此提交保留提供文件中的两个数值字段，记录来源摘要，不公开 `pre_cap` 原始层或内部采集路径。数据结构、覆盖范围和封顶计算已核对，采集 trace、参考实现等价性及 NPU 运行验收仍需另外复核。

新提供的 A3 文件含 MX/Hash 折算数据以及 5 个零值代理 case，未在本次 950 数据更新中替换现有 910C 文件。A3 不支持项仍按前述规则排除。

## 使用和验收

现有评测入口默认读取 `tasks/`。复现旧题集时显式指定 `--task-dir archives/season1/tasks`。Season1 与 Season2 题目和用例不同，成绩应分别标记，不直接混排。

公开前验证文件数量、公开 case ID、YAML/CSV 一致性、等级一致性、Golden 语法和入口、metadata 引用以及原题集归档完整性，并对新增的索引校验契约执行 CPU 回归测试。NPU 编译、全量精度和性能复测属于发布验收，不由上述静态检查替代。
