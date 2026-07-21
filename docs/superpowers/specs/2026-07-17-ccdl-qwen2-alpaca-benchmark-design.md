# CCDL Qwen2-0.5B Alpaca 双卡训练基准设计

## 目标

在 `wangjun@192.168.1.100` 的两张 RTX 4090 D 上，以相同模型、数据、精度、批量与 token 预算，对比 NCCL FP32 梯度同步和 CCDL INT8/INT4（Top-K 0/2）同步，测量通信加速、端到端吞吐、验证集质量、达到基线质量目标的步数和墙钟时间。

## 工作负载

- 从 `/home/wangjun/work/vlm_models/llava-onevision-qwen2-0.5b-ov-hf` 提取 `language_model.*` 权重，构造 `Qwen2ForCausalLM`。
- 使用同目录 tokenizer；输入数据为 `/home/wangjun/work/DeepSpeedExamples-master/training/tensor_parallel/alpaca_data.json`。
- 采用固定种子的 95%/5% 训练/验证划分。格式为 instruction、可选 input、response；只在 response token 上计算训练损失。
- 全参数 BF16 前后向，通信前将梯度展平为 FP32；基线使用 `dist.all_reduce`，压缩组使用 CCDL `qall_reduce`。
- 固定最大序列长度 256、每卡 micro-batch 2、梯度累积 4；所有组使用相同 global token/sample 顺序。

## 实验矩阵

主矩阵为 `nccl_fp32`、`int8_k0`、`int8_k2`、`int4_k0`、`int4_k2`，每组使用种子 17、29、43，共 15 次训练。正式训练前依次执行权重提取验证、单卡前向、双卡同步、每组短跑。

训练预算以预检测得的速度和收敛曲线确定，但所有配置必须采用完全相同的优化步数、评估间隔与样本顺序。不得因某个量化配置提前达标而提前停止。

## 指标与收敛

- 通信：平均/中位/p95 同步时间、压缩通信字节估计、相对 FP32 加速比、量化相对 L2 误差。
- 训练：tokens/s、samples/s、step time、端到端墙钟时间、峰值显存。
- 质量：验证 loss、perplexity、response-token accuracy。
- 收敛步数：先汇总三个 FP32 种子的最终 perplexity 均值，目标设为其 1.01 倍；每次运行首次连续 3 个评估点不高于目标的 step 为收敛 step。若未达到则明确标为未收敛，不外推。
- 墙钟收敛时间：从训练开始到收敛 step 的累计真实时间。

## 可复现性与产物

记录主机、GPU、镜像、PyTorch/Transformers/NCCL/CCDL 版本、模型与数据哈希、完整参数、每步/评估 JSONL、异常及退出码。结果输出至远端独立 run 目录，完成后生成 JSON/CSV、中文 Markdown 报告和 SVG 图，并下载到本地 `benchmark-results/`。

任何正式结论只使用真实双卡运行数据；smoke、synthetic 和失败重试不计入主矩阵。
