#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# CANN-Bench 报告生成脚本
#
# 用法:
#   ./scripts/utils/gen_report.sh eval_xxx.json                          # 从 JSON 生成 HTML
#   ./scripts/utils/gen_report.sh --json eval_xxx.json                   # 同上
#   ./scripts/utils/gen_report.sh --json eval_xxx.json --template custom/index.html  # 指定摘要模板
#   ./scripts/utils/gen_report.sh --help
#
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_DIR="${PROJECT_ROOT}/src"
DEFAULT_TEMPLATE="${PROJECT_ROOT}/tasks/description.html"

# 参数解析
JSON_FILE=""
TEMPLATE_FILE=""
HELP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --json|-j)
            JSON_FILE="$2"; shift 2 ;;
        --template|-t)
            TEMPLATE_FILE="$2"; shift 2 ;;
        --help|-h)
            HELP=1; shift ;;
        -*)
            echo "未知选项: $1"; shift ;;
        *)
            JSON_FILE="$1"; shift ;;
    esac
done

if [ -n "$HELP" ]; then
    echo "用法: $0 [--json|-j] <eval_xxx.json> [--template|-t] <index.html>"
    echo ""
    echo "选项:"
    echo "  --json, -j <path>    评测结果 JSON 文件路径（必需）"
    echo "  --template, -t <path> 摘要 + Section1 模板（默认: tasks/description.html）"
    echo "  --help, -h           显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 eval_20260601_114121.json"
    echo "  $0 --json reports/cann/eval_xxx.json --template tasks/description.html"
    exit 0
fi

if [ -z "$JSON_FILE" ]; then
    echo "错误: 请提供 JSON 文件路径"
    echo "用法: $0 [--json] <eval_xxx.json>"
    exit 1
fi

if [ ! -f "$JSON_FILE" ]; then
    echo "错误: JSON 文件不存在: $JSON_FILE"
    exit 1
fi

TEMPLATE_FILE="${TEMPLATE_FILE:-$DEFAULT_TEMPLATE}"

echo "=============================================="
echo "CANN-Bench 报告生成"
echo "=============================================="
echo "JSON 文件:   $JSON_FILE"
echo "模板文件:    $TEMPLATE_FILE"
echo "=============================================="

PYTHONPATH="${SRC_DIR}:${PYTHONPATH}" python3 -c "
import sys, json, os
sys.path.insert(0, sys.argv[1])
json_file = sys.argv[2]
template_file = sys.argv[3]

from kernel_eval.report.html_generator import render_html_report
from kernel_eval.report.report_generator import EvalReport, EvalResult, OperatorReport

# 读取 JSON
with open(json_file) as f:
    data = json.load(f)

# 构造 OperatorReport 列表
operators = []
for op_data in data.get('operators', []):
    op = OperatorReport(
        rel_path=op_data.get('rel_path', ''),
        operator=op_data.get('operator', ''),
        total_cases=op_data.get('total_cases', 0),
        passed_cases=op_data.get('passed_cases', 0),
        failed_cases=op_data.get('failed_cases', 0),
        pass_rate=op_data.get('pass_rate', 0.0),
        avg_speedup=op_data.get('avg_speedup', 0.0),
        score=op_data.get('score', 0.0),
        cases=[EvalResult(**case_data) for case_data in op_data.get('cases', [])],
    )
    operators.append(op)

# 构造 EvalReport
report = EvalReport(
    framework_version=data.get('framework_version', data.get('version', '')),
    tasks_version=data.get('tasks_version', ''),
    eval_code=data.get('eval_code', ''),
    timestamp=data.get('timestamp', ''),
    device=data.get('device', ''),
    total_operators=data.get('total_operators', len(operators)),
    total_cases=data.get('total_cases', 0),
    passed_cases=data.get('passed_cases', 0),
    failed_cases=data.get('failed_cases', 0),
    overall_score=data.get('overall_score', 0.0),
    operators=operators,
    summary=data.get('summary', {}),
    setup_info=data.get('setup_info', {}),
)

# 生成 HTML
html = render_html_report(report, report.setup_info, template_file)

# 输出到原 JSON 同目录
out_name = os.path.splitext(os.path.basename(json_file))[0] + '.html'
out_dir = os.path.dirname(os.path.abspath(json_file))
out_path = os.path.join(out_dir, out_name)

with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f'[INFO] 评测报告已保存到: {out_path}')
" "$SRC_DIR" "$JSON_FILE" "$TEMPLATE_FILE"

echo ""
echo "报告生成完成"
