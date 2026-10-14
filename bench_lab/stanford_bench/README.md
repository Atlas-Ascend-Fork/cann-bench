# StanfordBench

StanfordBench（原 KernelBench）是由 Scaling Intelligence 发布的算子评测基准数据集。

## 下载

```bash
bash bench_lab/stanford_bench/download.sh
```

数据将下载到 `bench_lab/stanford_bench/KernelBench/` 目录。

> 目录名 `KernelBench` 为 GitHub 仓库原名，评测时使用 `--bench-name stanford`。

## 数据来源

- 仓库: https://github.com/ScalingIntelligence/KernelBench
- 锁定 commit: `21fbe5a642898cd60b8f60c7aefb43d475e11f33`

## 评测使用

```bash
# 通过 run_evaluation.sh 自动下载并评测
bash scripts/run_evaluation.sh --bench-name stanford

# 或手动下载后指定路径
bash scripts/run_evaluation.sh --bench-name stanford --task-dir bench_lab/stanford_bench/KernelBench/KernelBench
```
