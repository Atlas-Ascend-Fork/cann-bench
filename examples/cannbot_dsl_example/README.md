# CANNBot DSL Example

基于 CANNBot-DSL 的 Ascend NPU 算子开发示例工程:使用纯 Python JIT 方式编写算子(无需 CMake 等构建系统),并打包为 `cann_bench` Python 包,供 CANN Bench 评测。

本样例用 CANNBot-DSL 实现 `add` / `sqrt` 两个轻量 elementwise 算子,用于演示 DSL 算子工程结构,并走通 cann-bench 评测流水线 smoke 验证(对接 cann-bench 仓库 `examples/tasks/` 下的 fixture 任务)。

## 目录结构

```text
cannbot_dsl_example/
├── build.sh            # 构建脚本,产出 dist/cann_bench-1.0.0-*.whl
├── setup.py            # 打包配置(find_packages 自动发现模块)
└── cann_bench/         # Python 包
    ├── __init__.py     # 导出算子接口(add、sqrt)
    ├── add.py          # add 算子(@host/@kernel JIT 实现)
    └── sqrt.py         # sqrt 算子(@host/@kernel JIT 实现)
```

## 前置条件

- Python 3.10 ~ 3.12
- Ascend 950PR / 950DT NPU
- 已安装 CANN toolkit,并在当前 shell 中 source 其环境变量脚本:

  ```bash
  source <cann安装路径>/set_env.sh
  ```

- 已安装 `torch` 与 `torch_npu`
- 已安装 `cannbot-dsl` 0.7.0:

  - 该 wheel 不在公共 pip 镜像上。
  - 请从 CANNBot-DSL CI 看板 <http://60.204.199.242/> 的 artifacts 目录下载对应版本的 wheel 文件,然后执行:

  ```bash
  pip install <下载的 cannbot_dsl wheel 文件路径>
  ```

## 构建方法

在工程根目录(`examples/cannbot_dsl_example`)执行:

```bash
bash build.sh
```

构建成功后,将在 `dist/` 目录下产出评测所需的 wheel 包:

```text
dist/cann_bench-1.0.0-*.whl
```

## 评测方法

在 cann-bench 仓库根目录执行:

```bash
./scripts/run_evaluation.sh --source-dir examples/cannbot_dsl_example --task-dir examples/tasks/level2/add --device-id 0
./scripts/run_evaluation.sh --source-dir examples/cannbot_dsl_example --task-dir examples/tasks/level1/sqrt --device-id 0
```

评测流程简述:

1. 调用工程内的 `build.sh` 构建 wheel(产出 `dist/cann_bench-1.0.0-*.whl`);
2. 安装该 wheel 到当前 Python 环境;
3. `import cann_bench`,按任务 schema 自动发现同名接口(如 `cann_bench.add`、`cann_bench.sqrt`);
4. 在指定 device 上运行参考实现与样例实现,进行精度对比与性能评分。

## 重要约束:host 侧不得依赖其他 kernel

**算子 host 侧代码不允许调用 torch 内置 kernel(如 `torch.nn.functional.pad`、`torch.zeros`、`copy_` 等)来做 GM 对齐 padding 或其他计算。** 这是 benchsite 的提交要求(算子必须自包含、不依赖其他 kernel 实现);同时 PadV3 等内置 kernel 在 runner 的 CANN 9.1.0 / 950PR 环境上没有对应 binary,调用会直接报错。

正确的做法是:host 侧不做任何 padding,直接把 `x.reshape(n)` 和 `out.reshape(n)` 传给 DSL kernel;尾块对齐由 DSL `tile_slice` 的 tail-aware 裁剪(最后一块不足整 tile 时自动按剩余长度搬移)配合 kernel 内的 `mask_counter` / `update_mask` 掩码处理。本工程两个样例算子均按此方式实现,可直接参考。

## 标准写法要点

`add.py` / `sqrt.py` 遵循以下标准实践,新增算子请照此实现:

1. **核数动态获取**:launch 的 block_dim 用 `cannbotdsl.get_platform_info().core_num` 在运行时查询(自动感知 stream 核数配额),不要写死常量。
2. **动态 Shape**:不要把 shape 作为 Python int 固化进 kernel(那会让每个 shape 触发一次重编译)。用 `cannbotdsl.compile(host_fn, TensorSpec((Dim("n"),), dtype), ...)` 走 AOT 路径,shape 以运行时参数传入,一份编译产物服务所有 shape。
3. **编译缓存**:内存层默认开启;在 import cannbotdsl 前设置 `CANNBOTDSL_CACHE_DIR`(样例默认 `~/.cache/cannbotdsl`)启用磁盘持久层,跨进程/跨次评测直接命中缓存。

## 新增算子步骤

以 `add` 为例,新增算子只需四步:

### 第一步:新建算子文件

在 `cann_bench/` 目录下新建算子文件,例如:

```text
cann_bench/<op>.py
```

### 第二步:实现 kernel 与 Python 接口

在该文件中用 CANNBot-DSL 的 `@host` / `@kernel` 装饰器实现 kernel 与 Python 接口。接口签名必须与对应任务的 proto.yaml 中 schema 一致。以 `add` 为例,其 schema 为:

```text
add(Tensor x1, Tensor x2) -> Tensor y
```

其中:

- `add` 表示 `x1` 与 `x2` 逐元素相加,返回单个张量 `y`;
- elementwise 算子无需 `epsilon` 等额外参数。

### 第三步:注册导出新算子

编辑 `cann_bench/__init__.py`,在该文件中导入并导出新算子接口(如 `add`、`sqrt`),使 `import cann_bench` 后可直接访问对应接口。

### 第四步:重新构建

在工程根目录执行:

```bash
bash build.sh
```

无需修改 `setup.py` 或 `build.sh`:`setup.py` 使用 `find_packages` 自动发现新模块,重新构建后即可将新算子打包进 wheel。
