#!/usr/bin/python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See the repository LICENSE for the full text.

"""CPU-only delivery validation; not NPU performance or admission approval.

Run with repository ``src`` and NumPy/PyTorch/PyYAML/pytest on sys.path.
Only the 20 cases in each operator's cases.yaml are public scoring candidates.
Synthetic semantic and mutation checks below are developer tests, not extra cases.
The loader/get_input path is real; full evaluator precision routing is NOT claimed.
"""

import ast
import csv
import inspect
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPOSITORY = Path(__file__).resolve().parents[2]
OPERATORS = REPOSITORY / "bench_lab" / "model_attention_bench"
sys.path.insert(0, str(REPOSITORY / "src"))

from kernel_eval.benches.cann_loader import CannCaseLoader, CannTaskLoader, GoldenLoader
from kernel_eval.data.data_generator import DataGenerator
from kernel_eval.eval.op_runner import OpRunner
from kernel_eval.utils.device_manager import DeviceConfig, DeviceManager
from kernel_eval.utils.param_builder import ParamBuilder

@pytest.fixture(scope="module", autouse=True)
def _bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)

NAMES = ("kimi_delta_attention", "compressed_sparse_attention_core")
TASKS = CannTaskLoader(str(OPERATORS))
CASES = CannCaseLoader(str(OPERATORS))
GOLDENS = GoldenLoader(str(OPERATORS))
GENERATOR = DataGenerator()
PARAMS = ParamBuilder()
TOLERANCES = {
    torch.float32: (1e-4, 1e-5),
    torch.float16: (3e-3, 3e-4),
    torch.bfloat16: (2e-2, 2e-3),
}


def _tuple(result):
    return result if isinstance(result, tuple) else (result,)


def _array(tensor):
    return tensor.detach().to(torch.float64).cpu().numpy().copy()


def _fp64(tensors):
    return [x.double() if x.is_floating_point() else x.clone() for x in tensors]


def _prepared(name, case_id, seed=20260908):
    """Use actual loaders and evaluator-style shared structural preprocessing."""
    task = TASKS.get_task(name)
    case = next(c for c in CASES.scan_by_rel_path(name) if c.case_num == case_id)
    tensors = GENERATOR.generate_input_tensors_from_case(
        case.input_shapes, case.dtypes, case.value_ranges, seed=seed + case_id
    )
    original = [x.clone() for x in tensors]
    hook = GOLDENS.get_input_function(name)
    if hook is not None:
        kwargs = {i.name: x for i, x in zip(task.inputs, tensors)}
        kwargs.update(case.attrs)
        kwargs.setdefault("skip2_exist", True)  # Same extra kwarg as evaluator.
        tensors = list(hook(**kwargs))
        again = list(hook(**kwargs))
        for x, y in zip(tensors, again):
            assert torch.equal(x, y), "get_input must be deterministic"
        for x, y in zip(kwargs.values(), original):
            if isinstance(x, torch.Tensor):
                assert torch.equal(x, y), "get_input must not mutate source tensors"
    else:
        assert name == "kimi_delta_attention"
    return task, case, tensors


def _call(function, case, tensors):
    kwargs = PARAMS.build_call_params(function, case, tensors)
    assert len(kwargs) == len(inspect.signature(function).parameters)
    return function(**kwargs)


def _kda_numpy(q, k, v, g, beta, state, normalize=True, scale=-1.0, mutation=None):
    """Independent explicit transition-matrix reference, only for tiny tensors.

    Unlike delivered Golden/oracle rank-one residual recurrences, construct
    M_t=(I-beta*k*k^T)@diag(exp(g)) explicitly for each batch/head/token.
    """
    q, k, v, g, beta, state = map(_array, (q, k, v, g, beta, state))
    b, t, h, dk = q.shape
    if normalize:
        q /= np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6)
        k /= np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    scale = dk ** -0.5 if scale <= 0 else scale
    y = np.empty(v.shape, dtype=np.float64)
    for bi in range(b):
        for hi in range(h):
            memory = state[bi, hi].copy()
            for ti in range(t):
                kt, vt = k[bi, ti, hi], v[bi, ti, hi]
                strength = beta[bi, ti, hi]
                decay = np.diag(np.exp(g[bi, ti, hi]))
                delta = np.eye(dk) - strength * np.outer(kt, kt)
                update = strength * np.outer(kt, vt)
                if mutation == "omit_state_update":
                    memory = decay @ memory
                elif mutation == "wrong_gate_axis":
                    memory = delta @ (memory @ decay) + update
                elif mutation == "decay_after_update":
                    memory = decay @ (delta @ memory + update)
                else:
                    memory = (delta @ decay) @ memory + update
                y[bi, ti, hi] = scale * q[bi, ti, hi] @ memory
            state[bi, hi] = memory
    return y, state


def _csa_numpy(query, compressed, window, indices, sink, query_start=0, scale=-1.0,
               mutation=None):
    """Independent per-query/per-head stable softmax, no Golden helpers."""
    query, compressed, window, sink = map(_array, (query, compressed, window, sink))
    indices = indices.cpu().numpy()
    b, s, h, d = query.shape
    scale = d ** -0.5 if scale <= 0 else scale
    origin = max(0, query_start - 127)
    result = np.empty_like(query)

    def attend(q, entries, log_sink):
        logits = entries @ q * scale
        if mutation == "omit_sink":
            peak = np.max(logits)
            weights = np.exp(logits - peak)
            return (weights @ entries) / weights.sum()
        peak = max(float(log_sink), float(np.max(logits)))
        weights = np.exp(logits - peak)
        return (weights @ entries) / (weights.sum() + np.exp(log_sink - peak))

    for bi in range(b):
        for si in range(s):
            t = query_start + si
            raw = window[bi, max(0, t - 127) - origin:t + 1 - origin]
            selected = [int(j) for j in indices[bi, si] if j >= 0]
            assert len(selected) == len(set(selected))
            assert all(4 * (j + 1) <= t + 1 for j in selected)
            comp = compressed[bi, selected]
            entries = np.concatenate((raw, comp), axis=0)
            for hi in range(h):
                if mutation == "two_softmaxes" and len(comp):
                    result[bi, si, hi] = (
                        attend(query[bi, si, hi], raw, sink[hi])
                        + attend(query[bi, si, hi], comp, sink[hi])
                    )
                else:
                    result[bi, si, hi] = attend(query[bi, si, hi], entries, sink[hi])
    return result


@pytest.mark.parametrize("name", NAMES)
def test_five_file_contract_and_real_loaders(name):
    folder = OPERATORS / name
    for filename in ("proto.yaml", "desc.md", "golden.py", "cases.yaml", "cases.csv"):
        assert (folder / filename).is_file()
    proto = yaml.safe_load((folder / "proto.yaml").read_text(encoding="utf-8"))["operator"]
    rows = yaml.safe_load((folder / "cases.yaml").read_text(encoding="utf-8"))["cases"]
    with (folder / "cases.csv").open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(rows) == len(csv_rows) == 20
    assert [r["case_id"] for r in rows] == list(range(1, 21))
    for row, exported in zip(rows, csv_rows):
        parsed = {key: ast.literal_eval(value) if key in {
            "case_id", "input_shape", "dtype", "attrs", "value_range"
        } else value for key, value in exported.items()}
        assert parsed == row
        assert row["operator"] == proto["name"]
        assert len(row["input_shape"]) == len(row["dtype"]) == len(proto["inputs"])
        assert set(row["attrs"]) == {a["name"] for a in proto["attrs"]}
        assert "baseline_perf_us" not in row and "t_hw_us" not in row
    task = TASKS.get_task(name)
    assert task is not None and task.name == proto["name"]
    assert len(CASES.scan_by_rel_path(name)) == 20
    function = GOLDENS.get_golden_function(name)
    oracle = GOLDENS.get_oracle_function(name, required=True)
    bench = GOLDENS.get_bench_function(name, required=True)
    expected = [i["name"] for i in proto["inputs"]] + [a["name"] for a in proto["attrs"]]
    for entry in (function, oracle, bench):
        signature = inspect.signature(entry)
        assert list(signature.parameters) == expected
        assert all("Tensor" in str(signature.parameters[i["name"]].annotation) for i in proto["inputs"])
        for attr in proto["attrs"]:
            assert signature.parameters[attr["name"]].default == attr["default"]
    desc = (folder / "desc.md").read_text(encoding="utf-8")
    embedded = re.search(r"```python\n(.*?)\n```", desc, re.DOTALL)
    assert embedded and embedded.group(1) == (folder / "golden.py").read_text(encoding="utf-8").rstrip("\n")


@pytest.mark.parametrize("seed", (20260908, 73291))
@pytest.mark.parametrize("case_id", range(1, 21))
@pytest.mark.parametrize("name", NAMES)
def test_all_public_cases_against_fp64_oracle(name, case_id, seed):
    task, case, tensors = _prepared(name, case_id, seed)
    before = [x.clone() for x in tensors]
    if name == "compressed_sparse_attention_core":
        ids = tensors[3]
        assert ids.dtype == torch.int32
        for si in range(ids.shape[1]):
            for row in ids[:, si]:
                valid = row[row >= 0]
                assert len(valid) == len(torch.unique(valid))
                assert torch.all((valid + 1) * 4 <= case.attrs["query_start"] + si + 1)
    actual = _tuple(_call(GOLDENS.get_golden_function(name), case, tensors))
    reference = _tuple(_call(GOLDENS.get_oracle_function(name, required=True), case, _fp64(tensors)))
    assert len(actual) == len(reference) == len(task.outputs)
    for i, (got, wanted) in enumerate(zip(actual, reference)):
        assert wanted.dtype == torch.float64, "oracle must not silently downcast"
        assert torch.isfinite(got).all() and torch.isfinite(wanted).all()
        expected_shape = tensors[2].shape if name == "kimi_delta_attention" else tensors[0].shape
        if i == 1:
            expected_shape = tensors[5].shape
        assert got.shape == wanted.shape == expected_shape
        assert got.dtype == (torch.float32 if i == 1 else tensors[0].dtype)
        rtol, atol = (2e-4, 2e-5) if i == 1 else TOLERANCES[tensors[0].dtype]
        torch.testing.assert_close(got.double(), wanted, rtol=rtol, atol=atol)
    for old, current in zip(before, tensors):
        assert torch.equal(old, current), "Golden must not mutate input tensors"


def _small_kda(length=7, dtype=torch.float64, dk=3, dv=3):
    rng = torch.Generator().manual_seed(143)
    q = torch.randn(2, length, 2, dk, generator=rng, dtype=dtype) * 0.2
    k = torch.randn(q.shape, generator=rng, dtype=dtype) * 0.2
    v = torch.randn(2, length, 2, dv, generator=rng, dtype=dtype) * 0.2
    g = -torch.rand(q.shape, generator=rng, dtype=dtype)
    beta = torch.rand(2, length, 2, generator=rng, dtype=dtype)
    state = torch.randn(2, 2, dk, dv, generator=rng, dtype=dtype) * 0.1
    return [q, k, v, g, beta, state]


@pytest.mark.parametrize("normalize", (True, False))
@pytest.mark.parametrize("scale", (-3.0, -1.0, 0.0, 0.7))
def test_kda_independent_matrix_reference(normalize, scale):
    data = _small_kda(dk=3, dv=5)
    expected = _kda_numpy(*data, normalize=normalize, scale=scale)
    for function in (GOLDENS.get_golden_function(NAMES[0]), GOLDENS.get_oracle_function(NAMES[0])):
        for got, wanted in zip(function(*data, normalize, scale), expected):
            assert got.dtype == torch.float64
            np.testing.assert_allclose(_array(got), wanted, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("cut", (63, 64, 65))
def test_kda_state_continuation(cut):
    data = _small_kda(length=131, dtype=torch.float32, dk=3, dv=5)
    function = GOLDENS.get_golden_function(NAMES[0])
    full_y, full_state = function(*data)
    first = [x[:, :cut] for x in data[:5]] + [data[5]]
    first_y, first_state = function(*first)
    second = [x[:, cut:] for x in data[:5]] + [first_state]
    second_y, second_state = function(*second)
    torch.testing.assert_close(torch.cat((first_y, second_y), dim=1), full_y, rtol=0, atol=0)
    torch.testing.assert_close(second_state, full_state, rtol=0, atol=0)


def test_kda_epsilon_and_zero_degeneracies():
    function = GOLDENS.get_golden_function(NAMES[0])
    q = torch.full((1, 1, 1, 1), 1e-4, dtype=torch.float64)
    data = [q, q.clone(), torch.ones_like(q), torch.zeros_like(q),
            torch.ones(1, 1, 1, dtype=torch.float64), torch.zeros(1, 1, 1, 1, dtype=torch.float64)]
    y, state = function(*data, True, 1.0)
    expected_key = 1e-4 / math.sqrt(1e-8 + 1e-6)
    np.testing.assert_allclose(_array(y), expected_key ** 2, rtol=1e-12)
    np.testing.assert_allclose(_array(state), expected_key, rtol=1e-12)
    for zero_index in (0, 1, 4):
        data = _small_kda()
        data[zero_index].zero_()
        result = function(*data)
        expected = _kda_numpy(*data)
        for got, wanted in zip(result, expected):
            np.testing.assert_allclose(_array(got), wanted, rtol=2e-12, atol=2e-12)
        if zero_index == 0:
            assert torch.count_nonzero(result[0]) == 0
        else:
            faded = data[5] * torch.exp(data[3].sum(dim=1)).unsqueeze(-1)
            torch.testing.assert_close(result[1], faded, rtol=2e-12, atol=2e-12)


def test_kda_scale_only_changes_output_not_state():
    data = _small_kda()
    function = GOLDENS.get_golden_function(NAMES[0])
    y1, state1 = function(*data, True, 0.2)
    y2, state2 = function(*data, True, 0.6)
    torch.testing.assert_close(y2, y1 * 3, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(state2, state1, rtol=0, atol=0)


@pytest.mark.parametrize("mutation", ("omit_state_update", "wrong_gate_axis", "decay_after_update"))
def test_kda_mutations_are_detected(mutation):
    data = _small_kda()
    expected = _kda_numpy(*data)
    mutant = _kda_numpy(*data, mutation=mutation)
    actual = GOLDENS.get_golden_function(NAMES[0])(*data)
    for got, wanted in zip(actual, expected):
        np.testing.assert_allclose(_array(got), wanted, rtol=2e-12, atol=2e-12)
    assert max(np.max(np.abs(a - b)) for a, b in zip(mutant, expected)) > 1e-3


def _small_csa(start=3, length=3):
    rng = torch.Generator().manual_seed(87)
    query = torch.randn(2, length, 3, 5, generator=rng, dtype=torch.float64) * 0.2
    compressed = torch.randn(2, (start + length) // 4, 5, generator=rng, dtype=torch.float64)
    window = torch.randn(2, length + min(start, 127), 5, generator=rng, dtype=torch.float64)
    ids = torch.full((2, length, 4), -1, dtype=torch.int32)
    for si in range(length):
        count = min(4, (start + si + 1) // 4)
        ids[:, si, :count] = torch.arange(count, dtype=torch.int32)
    sink = torch.tensor([-2.0, 0.2, 3.0], dtype=torch.float64)
    return [query, compressed, window, ids, sink]


@pytest.mark.parametrize("start", (0, 2, 3, 127, 128, 131))
@pytest.mark.parametrize("scale", (-2.0, 0.0, 0.5))
def test_csa_independent_per_head_reference(start, scale):
    data = _small_csa(start)
    expected = _csa_numpy(*data, query_start=start, scale=scale)
    actual = GOLDENS.get_oracle_function(NAMES[1])(*data, start, scale)
    assert actual.dtype == torch.float64
    np.testing.assert_allclose(_array(actual), expected, rtol=2e-12, atol=2e-12)


def test_csa_first_block_padding_and_joint_denominator():
    function = GOLDENS.get_golden_function(NAMES[1])
    # At t=3 all four raw entries AND compressed block 0 remain separate.
    q = torch.zeros(1, 1, 1, 1)
    comp = torch.tensor([[[2.0]]])
    raw = torch.ones(1, 4, 1)
    sink = torch.zeros(1)
    ids = torch.tensor([[[0, -1, -1]]], dtype=torch.int32)
    result = function(q, comp, raw, ids, sink, 3, 1.0)
    torch.testing.assert_close(result, torch.ones_like(result), rtol=1e-6, atol=1e-6)
    # 4*1 + 2 divided by 4 raw + 1 compressed + 1 sink = 1.
    window_only = function(q, comp, raw, torch.full_like(ids, -1), sink, 3, 1.0)
    torch.testing.assert_close(window_only, torch.full_like(result, 4 / 5), rtol=1e-6, atol=1e-6)
    assert not torch.equal(result, window_only), "first completed block must be visible"


def test_csa_window_eviction_at_127_128():
    function = GOLDENS.get_golden_function(NAMES[1])
    q = torch.zeros(1, 2, 1, 1)
    comp = torch.zeros(1, 32, 1)
    raw = torch.zeros(1, 129, 1)
    raw[:, 0] = 1.0
    ids = torch.full((1, 2, 1), -1, dtype=torch.int32)
    output = function(q, comp, raw, ids, torch.zeros(1), 127, 1.0)
    expected = torch.tensor([1 / 129, 0.0]).reshape_as(output)
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-8)


def test_csa_sink_dominance_and_large_logits_are_stable():
    function = GOLDENS.get_golden_function(NAMES[1])
    q = torch.zeros(1, 1, 1, 16)
    comp = torch.empty(1, 0, 16)
    raw = torch.ones(1, 1, 16)
    ids = torch.full((1, 1, 1), -1, dtype=torch.int32)
    sink = torch.tensor([20.0])  # Finite legal-domain upper bound.
    output = function(q, comp, raw, ids, sink, 0, 1.0)
    expected = torch.full_like(output, 1 / (1 + math.exp(20)))
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-12)
    output = function(q + 8, comp, raw * 8, ids, sink, 0, 1.0)
    assert torch.isfinite(output).all()  # Dot logit=1024, naive exp overflows.
    torch.testing.assert_close(output, torch.full_like(output, 8), rtol=0, atol=0)


@pytest.mark.parametrize("mutation", ("omit_sink", "two_softmaxes"))
def test_csa_mutations_are_detected(mutation):
    data = _small_csa(3)
    expected = _csa_numpy(*data, query_start=3)
    mutant = _csa_numpy(*data, query_start=3, mutation=mutation)
    actual = GOLDENS.get_oracle_function(NAMES[1])(*data, 3)
    np.testing.assert_allclose(_array(actual), expected, rtol=2e-12, atol=2e-12)
    assert np.max(np.abs(mutant - expected)) > 1e-2


@pytest.mark.parametrize(
    ("input_dtype", "expected_state_dtype"),
    (
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
    ),
)
def test_kda_oracle_final_state_dtype_matches_contract(input_dtype, expected_state_dtype):
    aux_dtype = torch.float64 if input_dtype == torch.float64 else torch.float32
    q = torch.ones((1, 1, 1, 2), dtype=input_dtype)
    k = torch.ones_like(q)
    v = torch.full((1, 1, 1, 3), 0.25, dtype=input_dtype)
    g = torch.zeros_like(q, dtype=aux_dtype)
    beta = torch.ones((1, 1, 1), dtype=aux_dtype)
    initial_state = torch.zeros((1, 1, 2, 3), dtype=aux_dtype)

    y, state = GOLDENS.get_oracle_function(NAMES[0])(
        q, k, v, g, beta, initial_state, False, 1.0
    )

    assert y.dtype == input_dtype
    assert state.dtype == expected_state_dtype
    assert state.shape == (1, 1, 2, 3)


def test_oracles_preserve_sub_fp32_information():
    # Dtype checks alone do not detect an FP32 compute followed by FP64 cast.
    q = torch.full((1, 1, 1, 1), 1.0, dtype=torch.float64)
    kda_data = [q, q.clone(), q + 2 ** -35, torch.zeros_like(q),
                torch.ones(1, 1, 1, dtype=torch.float64), torch.zeros_like(q)]
    y, state = GOLDENS.get_oracle_function(NAMES[0])(*kda_data, False, 1.0)
    assert y.dtype == state.dtype == torch.float64
    assert y.item() == state.item() == 1 + 2 ** -35
    csa_data = [q * 0, torch.empty(1, 0, 1, dtype=torch.float64),
                torch.tensor([[[1 + 2 ** -35]]], dtype=torch.float64),
                torch.full((1, 1, 1), -1, dtype=torch.int32), torch.zeros(1, dtype=torch.float64)]
    result = GOLDENS.get_oracle_function(NAMES[1])(*csa_data, 0, 1.0)
    assert result.dtype == torch.float64
    assert result.item() == (1 + 2 ** -35) / 2


@pytest.mark.parametrize("name", NAMES)
def test_real_oprunner_rebinds_low_precision_params_to_fp64_inputs(name):
    """Regression for evaluator get_input -> OpRunner precision handoff.

    Low-precision params are intentional: the real runner replaces Tensor
    references from its golden_inputs list before invoking the oracle, including
    on the CSA get_input path. Integer structural inputs must remain integers.
    """
    task, case, tensors = _prepared(name, 1)
    assert tensors[0].dtype in (torch.float16, torch.bfloat16)
    function = GOLDENS.get_golden_function(name)
    oracle = GOLDENS.get_oracle_function(name, required=True)
    low_precision_params = PARAMS.build_call_params(function, case, tensors)
    golden_inputs = _fp64(tensors)
    captured = {}

    def spy_oracle(**kwargs):
        captured.update(kwargs)
        for index, input_info in enumerate(task.inputs):
            received = kwargs[input_info.name]
            assert received is golden_inputs[index]
            if tensors[index].is_floating_point():
                assert received.dtype == torch.float64
            else:
                assert received.dtype == tensors[index].dtype == torch.int32
                assert torch.equal(received, tensors[index])
        return oracle(**kwargs)

    runner = OpRunner(DeviceManager(DeviceConfig(type="cpu")))
    run = runner.run(
        spy_oracle, low_precision_params, case.case_id, golden_inputs,
        to_device=False, enable_profiler=False,
    )
    assert run.success, (run.error, run.traceback)
    assert run.device == "cpu" and run.perf_result is None
    assert captured, "the real runner must actually invoke the spy oracle"
    direct = _tuple(_call(oracle, case, golden_inputs))
    for got, wanted in zip(_tuple(run.outputs), direct):
        assert got.dtype == wanted.dtype == torch.float64
        torch.testing.assert_close(got, wanted, rtol=0, atol=0)
    # Rebinding is non-mutating: caller's original parameter dictionary survives.
    for index, input_info in enumerate(task.inputs):
        assert low_precision_params[input_info.name] is tensors[index]
