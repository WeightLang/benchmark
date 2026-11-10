# Weight Initialization Benchmark

这个 benchmark 让 **AI agent 编写模型权重初始化算法**，再将算法产生的初始权重
与高斯、Xavier、Kaiming 等随机初始化组做配对比较。目标是衡量 AI 对不同通用
模型架构及其权重作用的理解：能否根据参数形状、连接关系、激活函数、残差路径、
注意力和归一化结构，设计有用的初始化规则，而不只是照搬一种随机分布。

题目固定，架构按种子抽取。被评测的 AI agent 读取 WGT 架构，编写可重复运行的
初始化算法，生成各架构的初始权重；评测器在相同数据与训练预算下运行候选组和
随机初始化对照组。**被评测的 AI agent** 与执行数字识别/字符预测的**下游小模型**
是两个不同角色。跨架构、跨种子的表现提升是架构理解能力的行为证据，不等同于
直接证明内部理解；应同时审查算法源码，检查收益是否只来自某个架构或任务。

使用 **Harbor + WeightLang 原生 Rust/wgtime** 执行这一流程。Harbor 负责 agent
环境与独立评分容器，WeightLang 负责模型架构及原生推理、训练。

| 固定任务 | 输入与目标 | 数据 |
|---|---|---|
| 手写数字识别 | 8×8 灰度 → 10 类 | UCI Optical Digits，非 MNIST |
| 小型语言模型 | 前 4 个字符 → 下一个字符，16 字符词表 | 项目原创固定语料，按整句划分训练/测试 |

模型包括 MLP、CNN、单残差块 ResNet、双头 Transformer。它们是 CPU 上可快速
重复的小模型，不是完整 ResNet-18 或 GPT。Transformer 对过去四个字符做注意力，
不读取预测目标；展平读出保留位置区别。CNN/ResNet 在字符任务上的输入是 one-hot
网格，属于跨架构对照，不代表推荐的语言模型设计。

## 复现

需要 Linux、Rust 1.99.0、C 编译器、Python 3、Docker 和 Harbor 0.25.0。
WeightLang 固定到 `WEIGHTLANG_REVISION`；SDK/daemon 使用该源码构建，不使用 PATH
上可能过时的 `wgt`。首次构建前自行检查目标目录与 Docker 存储空间。

```sh
./scripts/bootstrap.sh
cargo test --locked
python3 -m unittest discover -s tests -v
docker build -t weight-init-bench:0.1.0 .
harbor run -p tasks/weight-init -a oracle -n 1
# 使用实际 agent；模型名按已配置服务填写：
harbor run -p tasks/weight-init -a codex -m MODEL -n 1
```

若 Docker Hub 的 Rust 大镜像下载停滞，而本机已有官方 Rust 1.99.0 工具链及 Cargo
registry 缓存，可以用 `./scripts/build-local-image.sh` 构建同名镜像。它在 Debian
容器内挂载编译器和只读依赖缓存编译，仍使用容器的 libc，不复制宿主机二进制。
agent 环境允许联网安装工具；独立 verifier 无网络。

不通过 Harbor 也可以验收相同的原生执行路径：

```sh
mkdir -p results
target/debug/weight-init-bench emit . xavier_uniform > results/candidate.json
target/debug/weight-init-bench evaluate . results/candidate.json results/report.json
# 只跑随机初始化对照：
target/debug/weight-init-bench baselines . results/baselines.json
```

报告默认拒绝覆盖。每次执行启动独立 Unix socket 的 wgtime 实例，退出后回收服务、
临时权重与编译缓存，不连接或重启机器上已有的服务。

## 实验协议 v0.1

- 固定两个任务，每项训练/测试各 96 个样本。测试数字与训练数字索引不重叠；
  语言数据先按整句划分，再构造窗口。原始出处、索引、SHA-256 见 `data/`。
- 从四类架构中按固定种子**无放回抽取**，抽完一轮才重新填池。默认抽四个，
  因此覆盖全部家族且顺序随机；目前不随机化家族内的宽度/深度。
- 每个任务/架构使用种子 11、29、47，共 24 个 trial。候选和对照完全配对。
- 三个随机对照：Normal(0, 0.05)、Xavier Uniform、Kaiming Normal（ReLU fan-in）。
  偏置为零、LayerNorm gamma 为一。卷积 fan 包含空间核，矩阵乘布局区别处理。
  每个参数通过种子和完整参数名派生独立 RNG。
- 所有方法统一 CPU、F32、Adam、学习率 0.01、20 步、batch 16、全局梯度裁剪 1。
  每步按独立固定数据流有放回采样，候选与对照使用同一序列。
- 每个实例从完整权重新建，优化器状态不共享。记录训练前/后准确率、交叉熵、
  训练损失曲线；语言任务另报告 perplexity。
- Harbor reward 是训练后测试准确率的平均值，两个任务权重相等。同时报告各任务
  结果和对每个对照的配对准确率差。不同任务的难度不同，应优先阅读分项结果。
- Oracle 输出 Xavier 权重，期望与 Xavier 对照逐项相等，配对差为零。
  **这是连续性能评分，Oracle 不要求 reward=1。** 空提交/非法值应失败并得零分。

任务要求 agent 提交可重复运行的初始化算法源码、设计说明及其生成的权重；可用
公开训练数据的输入统计量设计规则，但不能通过拟合任务标签或预训练来替代初始化。
算法应适用于抽取的架构，不应靠 trial ID 查表或手工填写每组答案。

当前自动评分器仅验证和运行权重数据，尚未在独立受限环境中重新执行算法源码，
因此不能自动证明权重由所交算法生成，也不能排除预训练或手工适配。源码和设计
说明暂供审查，当前得分只能作为理解能力的初步代理指标，不能称为已验证的
“初始化算法理解能力排名”。后续严格赛道需要受限生成阶段、统一生成预算及
未见过的架构/种子测试，并明确验证实际来源。

24 个 trial 的标准差只描述本次固定样本，不是总体置信区间。96 个样本和短语料是
初始化工程验收规模，不足以支持大模型结论；公开 holdout 也不具备防污染保证。
未使用 GPU，不宣称 GPU 结果或大语言模型训练能力。

## 目录与修改

```text
models/*/model.wgt        四类固定架构的模板，运行时绑定 10/16 分类头
src/main.rs              Rust 调度、初始化对照、原生训练、评分
config/benchmark.json    固定预算与种子
data/                    数据、来源、划分与哈希
scripts/prepare_data.py  重建固定数据
scripts/bootstrap.sh     获取锁定 WeightLang 并构建
scripts/sync_task.py      同步 agent/verifier 的独立构建输入
tasks/weight-init/        Harbor 任务、参考解、独立评分容器
tests/                   协议、数据隔离和拒绝无效提交的测试
```

修改数据或预算后先运行 `python3 scripts/prepare_data.py`，再运行 bootstrap 同步。
模型/协议变更后也必须重新构建 Docker 基础镜像。生成的 Harbor 输入纳入版本控制，
可直接审查测试数据有没有进入 agent 镜像。公开仓库为
[WeightLang/benchmark](https://github.com/WeightLang/benchmark)，本仓库不自动触发 CI。

任务创建依据：[Harbor 官方 create-task 技能](https://github.com/harbor-framework/harbor/blob/main/skills/create-task/SKILL.md)。

本机测试和 Harbor Oracle/NOP 的实际验收结果见 [validation/README.md](validation/README.md)。
