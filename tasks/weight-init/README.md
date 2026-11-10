# weightlang/weight-init

Agent 为固定手写数字识别和字符语言模型任务提交初始权重，完整要求见
[instruction.md](instruction.md)。抽样架构：MLP/CNN/ResNet/Transformer；24 个试验。

环境使用本项目 `weight-init-bench:0.1.0`，基于 Debian Bookworm，包含 Rust 原生
runner、wgtime、C 编译器、Python 3。CPU 2 核、RAM 2 GiB，agent 超时 1200 秒，
agent 环境允许网络以安装工具；verifier 无网络。数字训练数据已缓存；语言语料为原创文本。

Verifier 在独立的干净容器中运行，仅从 `/logs/artifacts/submission.json` 读取有限
数值权重。固定架构和测试集来自 verifier 自己的镜像。程序验证完整参数绑定，随后
执行 20 步 Adam 和三种随机初始化对照。reward 为平均最终准确率；更多指标保存在
`/logs/verifier/report.json`。Oracle 是 Xavier 对照，不保证准确率为 1。

```text
instruction.md         agent 任务说明
task.toml              资源、超时、独立 verifier 配置
environment/           公共模型/manifest/训练集，agent 构建上下文
tests/                 固定评分脚本、模型与训练/测试集，verifier 构建上下文
solution/solve.sh      生成 Xavier 初始化的参考解
```

在项目根目录构建基础镜像后运行：

```sh
harbor run -p tasks/weight-init -a oracle -n 1
harbor run -p tasks/weight-init -a codex -m MODEL -n 1
```

模型、数据、评分协议及复现限制见[项目说明](../../README.md)。
