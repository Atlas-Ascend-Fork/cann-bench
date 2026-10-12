#!/usr/bin/env python3
# coding=utf-8
"""
scatter_block_update 算子性能采集模板

基于 cann-bench 性能采集规则:
1. 使用 torch_npu.profiler 硬件级计时（非墙钟）
2. 测量前执行 MatMul 升频 + ReduceMax 清 L2 cache
3. profiler schedule: warmup=3, active=5
4. 从 kernel_details.csv 解析，按 Step 分组 → 每 kernel 跨步取中位数 → 累加
5. 自动过滤升频/清cache 的 warmup kernel

运行方式:
    python3 test_baseline_perf.py 1          # 运行 case 1
    python3 test_baseline_perf.py all         # 运行全部 cases
    python3 test_baseline_perf.py all --csv   # 输出 CSV 格式结果
"""

import os
import sys
import csv
import json
import shutil
import tempfile
import logging
import random
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import cann_bench  # 触发 torch.ops.cann_bench / torch.ops.custom 算子注册

_logger = logging.getLogger(__name__)

# =============================================================================
# 参数配置
# =============================================================================
DEVICE_ID = int(os.environ.get("DEVICE_ID", "0"))
PROFILER_WARMUP = 3
PROFILER_REPEAT = 5
SEED = 42

CASES_CSV = "./cases.csv"

FREQ_BOOST_DIM = 10240
CACHE_CLEAR_SHAPE = (96, 1024, 1024)

WARMUP_KERNEL_SHAPES = {
    "10240,10240;10240,10240",
    "96,1024,1024;3",
}


# =============================================================================
# dtype 映射
# =============================================================================
DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
    "int8": torch.int8,
}


# =============================================================================
# 用例加载
# =============================================================================
def load_cases(csv_path: str) -> Dict[int, dict]:
    """从 cases.csv 加载测试用例。"""
    cases = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = int(row["case_id"])
            shapes = json.loads(row["input_shape"])
            dtypes = json.loads(row["dtype"])
            value_range = json.loads(row["value_range"])
            cases[case_id] = {
                "shapes": shapes,
                "dtypes": [DTYPE_MAP[d] for d in dtypes],
                "value_range": value_range,
                "note": row.get("note", ""),
            }
    return cases


def _generate_tensor(shape: List[int], dtype: torch.dtype,
                     value_range: List[Any], device: str) -> torch.Tensor:
    """根据 dtype 生成对应随机张量。"""
    lo, hi = value_range[0], value_range[1]
    if dtype in (torch.int32, torch.int64, torch.int8):
        lo, hi = int(lo), int(hi)
        return torch.randint(lo, hi + 1, shape, dtype=dtype).to(device)
    else:
        lo, hi = float(lo), float(hi)
        t = torch.rand(shape, dtype=torch.float32) * (hi - lo) + lo
        return t.to(dtype).to(device)


def build_inputs(case_cfg: dict, device: str):
    """根据 case 配置构造输入张量。"""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    shapes = case_cfg["shapes"]
    dtypes = case_cfg["dtypes"]
    value_range = case_cfg["value_range"]

    input_tensor = _generate_tensor(shapes[0], dtypes[0], value_range[0], device)
    indices_raw = _generate_tensor(shapes[1], dtypes[1], value_range[1], device)
    update_tensor = _generate_tensor(shapes[2], dtypes[2], value_range[2], device)

    # golden 的 get_input 会重排 indices 为规则索引：col0 = // bs, col1 = % bs
    bn, bs, D = input_tensor.shape
    T = indices_raw.shape[0]
    indices_flat = torch.arange(T, dtype=torch.int32, device=device)
    col1 = indices_flat // bs
    col2 = indices_flat % bs
    indices = torch.stack([col1, col2], dim=1)

    return input_tensor, indices, update_tensor


# =============================================================================
# NPU 升频 + L2 cache 清空
# =============================================================================
def boost_freq_and_clear_cache(device: str):
    mm1 = torch.rand((FREQ_BOOST_DIM, FREQ_BOOST_DIM), dtype=torch.float16).to(device)
    mm2 = torch.rand((FREQ_BOOST_DIM, FREQ_BOOST_DIM), dtype=torch.float16).to(device)
    reduce_input = torch.rand(CACHE_CLEAR_SHAPE, dtype=torch.float16).to(device)

    torch.matmul(mm1, mm2)
    torch.npu.synchronize()
    torch.max(reduce_input)
    torch.npu.synchronize()

    del mm1, mm2, reduce_input


def clear_cache(device: str):
    reduce_input = torch.rand(CACHE_CLEAR_SHAPE, dtype=torch.float16).to(device)
    torch.max(reduce_input)
    torch.npu.synchronize()
    del reduce_input


# =============================================================================
# Profiler 采集
# =============================================================================
def run_profiled(fn, prof_dir: str, device: str,
                 warmup: int = PROFILER_WARMUP,
                 repeat: int = PROFILER_REPEAT) -> str:
    import torch_npu

    os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
    os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '3'
    for name in ['', 'torch', 'torch_npu', 'torch_npu.profiler', 'ascend', 'profiler']:
        lg = logging.getLogger(name)
        lg.setLevel(logging.ERROR)
        lg.handlers = []
        lg.addHandler(logging.NullHandler())

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    )

    fn()
    boost_freq_and_clear_cache(device)

    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    sink_file = tempfile.NamedTemporaryFile(
        mode='w+', prefix='perf_profiler_', suffix='.log', delete=False
    )
    sink_fd = sink_file.fileno()

    try:
        os.dup2(sink_fd, 1)
        os.dup2(sink_fd, 2)

        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu.profiler.schedule(
                wait=0, warmup=warmup, active=repeat, repeat=1
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            experimental_config=experimental_config,
        ) as prof:
            for i in range(warmup + repeat):
                if i >= warmup:
                    clear_cache(device)
                fn()
                prof.step()

        try:
            from torch_npu.profiler.analysis.prof_common_func._multi_process_pool import MultiProcessPool
            pool = MultiProcessPool()
            pool.close_pool(wait=True)
        except Exception:
            pass

    finally:
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        sink_file.close()
        try:
            os.unlink(sink_file.name)
        except OSError:
            pass

    return prof_dir


# =============================================================================
# kernel_details.csv 解析
# =============================================================================
def locate_kernel_details_csv(prof_dir: str) -> Optional[str]:
    direct = os.path.join(prof_dir, "kernel_details.csv")
    if os.path.isfile(direct):
        return direct

    try:
        for entry in os.listdir(prof_dir):
            candidate = os.path.join(prof_dir, entry, "kernel_details.csv")
            if os.path.isfile(candidate):
                return candidate
    except OSError:
        pass

    for root, dirs, files in os.walk(prof_dir):
        if "kernel_details.csv" in files:
            return os.path.join(root, "kernel_details.csv")

    return None


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 1:
        return sorted_vals[n // 2]
    return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2


def parse_kernel_details(csv_path: str) -> Dict[str, Any]:
    step_kernel_times: Dict[str, Dict[str, List[float]]] = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step_id = row.get("Step Id", "").strip()
            duration_str = row.get("Duration(us)", "").strip()
            input_shapes = row.get("Input Shapes", "").strip()
            name = row.get("Name", "").strip()

            if not step_id or not duration_str:
                continue
            try:
                duration = float(duration_str)
            except ValueError:
                continue
            if duration <= 0:
                continue
            if input_shapes in WARMUP_KERNEL_SHAPES:
                continue

            if step_id not in step_kernel_times:
                step_kernel_times[step_id] = {}
            if name not in step_kernel_times[step_id]:
                step_kernel_times[step_id][name] = []
            step_kernel_times[step_id][name].append(duration)

    all_kernel_times: Dict[str, List[float]] = {}
    for step_id, kernels in step_kernel_times.items():
        for name, durations in kernels.items():
            step_sum = sum(durations)
            if name not in all_kernel_times:
                all_kernel_times[name] = []
            all_kernel_times[name].append(step_sum)

    device_kernels = {name: round(_median(times), 2) for name, times in all_kernel_times.items()}
    total_kernel_us = round(sum(device_kernels.values()), 2)

    return {
        "total_kernel_us": total_kernel_us,
        "device_kernels": device_kernels,
        "step_count": len(step_kernel_times),
    }


# =============================================================================
# 单 case 评测
# =============================================================================
def run_single_case(case_id: int, case_cfg: dict, device: str) -> Optional[Dict[str, Any]]:
    input_tensor, indices, update_tensor = build_inputs(case_cfg, device)

    input_shape = list(input_tensor.shape)
    indices_shape = list(indices.shape)
    update_shape = list(update_tensor.shape)
    dtype_str = str(input_tensor.dtype)

    print(f"\n{'='*60}")
    print(f"[CASE {case_id}] input_shape={input_shape}, indices_shape={indices_shape}, update_shape={update_shape}, dtype={dtype_str}")
    print(f"{'='*60}")

    # scatter_block_update 是 in-place 算子，每次调用会修改 input。
    # 为保持多次调用幂等，在 fn 内部 clone input；测得时间包含 clone 开销。
    def _run_op():
        return cann_bench.scatter_block_update(
            input_tensor.clone(), indices, update_tensor
        )

    prof_dir = tempfile.mkdtemp(prefix=f"perf_case{case_id}_")

    try:
        run_profiled(_run_op, prof_dir, device)

        csv_path = locate_kernel_details_csv(prof_dir)
        if csv_path is None:
            print(f"[ERROR] Case {case_id}: kernel_details.csv 未找到")
            return None

        metrics = parse_kernel_details(csv_path)
        elapsed_us = metrics["total_kernel_us"]

        print(f"[RESULT] Case {case_id}: elapsed_us={elapsed_us:.2f} us "
              f"(kernel median sum, {metrics['step_count']} steps)")

        sorted_kernels = sorted(metrics["device_kernels"].items(),
                                key=lambda x: x[1], reverse=True)
        for i, (name, us) in enumerate(sorted_kernels[:5]):
            pct = us / elapsed_us * 100 if elapsed_us > 0 else 0
            print(f"         [{i+1}] {name}: {us:.2f} us ({pct:.1f}%)")

        return {
            "case_id": case_id,
            "input_shape": input_shape,
            "indices_shape": indices_shape,
            "update_shape": update_shape,
            "dtype": dtype_str,
            "elapsed_us": elapsed_us,
            "device_kernels": metrics["device_kernels"],
            "step_count": metrics["step_count"],
            "note": case_cfg.get("note", ""),
        }

    except Exception as e:
        print(f"[ERROR] Case {case_id} 评测失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        shutil.rmtree(prof_dir, ignore_errors=True)


# =============================================================================
# 主函数
# =============================================================================
def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <case_id | all> [--csv]")
        print(f"       case_id: cases.csv 中的 case 编号")
        print(f"       all: 运行全部 cases")
        print(f"       --csv: 输出 CSV 格式结果到 stdout")
        sys.exit(1)

    arg = sys.argv[1]
    output_csv = "--csv" in sys.argv

    if not os.path.exists(CASES_CSV):
        print(f"[ERROR] cases.csv 不存在: {CASES_CSV}")
        sys.exit(1)
    cases = load_cases(CASES_CSV)

    torch.npu.set_device(DEVICE_ID)
    device = f"npu:{DEVICE_ID}"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Profiler: warmup={PROFILER_WARMUP}, repeat={PROFILER_REPEAT}")
    print(f"[INFO] 采集方式: torch_npu.profiler → kernel_details.csv → 中位数统计")

    if arg.lower() == "all":
        case_ids = sorted(cases.keys())
    else:
        try:
            case_ids = [int(arg)]
        except ValueError:
            print(f"[ERROR] 无效的 case_id: {arg}")
            sys.exit(1)

    results = []
    for cid in case_ids:
        if cid not in cases:
            print(f"[WARN] Case {cid} 不存在，跳过。")
            continue
        result = run_single_case(cid, cases[cid], device)
        if result:
            results.append(result)

    print(f"\n{'='*60}")
    print(f"[SUMMARY] 成功: {len(results)} / {len(case_ids)}")
    print(f"{'='*60}")

    if results:
        print(f"\n{'case_id':>8} {'input_shape':>28} {'indices_shape':>20} {'update_shape':>20} {'dtype':>12} {'elapsed_us':>12} {'note'}")
        print("-" * 120)
        for r in results:
            print(f"{r['case_id']:>8} {str(r['input_shape']):>28} {str(r['indices_shape']):>20} "
                  f"{str(r['update_shape']):>20} {r['dtype']:>12} {r['elapsed_us']:>12.2f} {r['note']}")

    if output_csv and results:
        print("\n--- CSV OUTPUT ---")
        print("case_id,input_shape,indices_shape,update_shape,dtype,elapsed_us,step_count,note")
        for r in results:
            print(f"{r['case_id']},\"{r['input_shape']}\",\"{r['indices_shape']}\",\"{r['update_shape']}\","
                  f"{r['dtype']},{r['elapsed_us']:.2f},{r['step_count']},{r['note']}")

    if results:
        output_json = os.path.join(os.path.dirname(__file__), "perf_results_scatter_block_update.json")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[INFO] 结果已保存到: {output_json}")


if __name__ == "__main__":
    main()
