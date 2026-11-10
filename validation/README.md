# 本地验收：2026-11-10

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
