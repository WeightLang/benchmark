# 本地验收记录

## SDK 源码同步：2026-11-20

本轮将 Rust SDK 固定到 WeightLang 功能提交
[072582a03a123b7d95c815d563689383033019d4](https://github.com/WeightLang/WeightLang/commit/072582a03a123b7d95c815d563689383033019d4)。
本机 vendor 由该已发布提交的 `git archive` 生成，785 个文件的 Git blob、文件模式及符号链接均逐项校验。
`scripts/bootstrap.sh` 仍按 `WEIGHTLANG_REVISION` 下载固定提交的归档；
vendor 以源码归档存放，来源由固定提交和逐文件内容校验确认。

本轮 Rust 4 项测试、Python 协议 4 项测试通过。Rust 测试覆盖四个模型族的解析、
参数合同和快照构造；协议测试确认可执行文件内嵌版本、vendor marker、源码 pin
和 Harbor 环境 manifest 一致。配套新 `wgtime` 还完成 6 次真实 `native-cpu`
初始化、快照保存/加载、前向评估及单步训练：ResNet × 数字/字符两项任务 ×
三个初始化方法，每次 batch=1，使用原有 96 个训练样本及 96 个测试样本。

构建与测试使用 Rust 1.100.0，并复用本机共享 target 和编译串行锁。
应用版本仍为 0.1.0，默认完整协议的 24 个 trial、20 步预算及数据均保持原值；
轻量验收仅在临时目录中设置单步预算。机器记录、实际命令、工具链、二进制及数据摘要见
[sdk-refresh-20261120.json](sdk-refresh-20261120.json)，原始运行结果见
[sdk-refresh-20261120-smoke.json](sdk-refresh-20261120-smoke.json)。

使用已经构建的匹配产物可复验该入口，输出路径必须是新文件：

```sh
python3 scripts/check_sdk_smoke.py \
  --binary target/debug/weight-init-bench \
  --daemon target/debug/wgtime \
  --report /tmp/weight-init-sdk-smoke.json
```

本轮证据覆盖 Rust 1.100.0 的本机 SDK 与轻量原生服务验收。
下面的 Rust 1.99.0、Harbor Oracle/NOP、完整训练预算及分数证据仍对应 2026-11-10
的旧源码提交，不能作为本轮重新运行的结果。

## 完整协议与 Harbor 验收：2026-11-10

Rust 4 项测试、Python 协议 4 项测试通过。96 次真实模型初始化/训练执行
（24 个 trial × 候选及 3 个对照）完成，没有使用模拟后端。

```sh
cargo test --locked
python3 -m unittest discover -s tests -v
./scripts/build-local-image.sh
harbor run -p tasks/weight-init -a oracle -n 1
harbor run -p tasks/weight-init -a nop -n 1
python3 scripts/check_oracle.py validation/oracle-report.json
```

Harbor 0.25.0 的独立 verifier 实际运行：Oracle reward=0.3172743055555555，
NOP reward=0，两个 Harbor trial 均无框架异常。Oracle 24 个 trial 的初始指标、
最终指标、训练损失曲线和权重摘要均与 Xavier 对照逐项相等。容器与本机原生执行
得分一致。Oracle 耗时约 93 秒（包含环境设置），NOP 约 75 秒。

| 初始化 | 数字最终准确率 | 字符最终准确率 | 字符困惑度 |
|---|---:|---:|---:|
| Normal(0, 0.05) | 31.68% | 26.65% | 10.06 |
| Xavier Uniform / Oracle | 34.11% | 29.34% | 10.37 |
| Kaiming Normal | 36.81% | 30.73% | 10.56 |

仅限本协议固定 96 样本、20 步 CPU 训练；不是正式大样本排名。没有运行收费模型
agent，没有运行 GPU，也没有发布到 Harbor registry。

原始指标在 `oracle-report.json`，构建版本与哈希在 `acceptance.json`。
Docker Hub 大 Rust 镜像下载停滞后，使用本机已安装的官方 Rust 1.99.0 工具链在
Debian 容器中编译。早期 Rust 1.91 编译不支持该 WeightLang 提交使用的
`atomic_try_update`，已将项目工具链统一固定为 1.99.0；未修改 WeightLang 源码。
