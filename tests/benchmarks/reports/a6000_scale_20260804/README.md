# A6000 单机与双机量化训练性能报告

## 结论

当前 INT8 `all_gather + async gather + async error-feedback` 路径在通信参与者
较少时收益明显，但不具备良好的 4/8 卡扩展性：

- 单机 2 卡比 PyTorch DDP 快 `1.22x`。
- 双机 2 卡在 1GbE TCP 上快 `1.92x`，说明压缩确实降低了跨机字节量。
- 单机 4 卡慢 `17.9%`，双机 4 卡慢 `3.0%`。
- 双机 8 卡仅为 DDP 吞吐的 `27.9%`，即慢约 `3.59x`。

因此，CCDL 当前 all-gather 路径适合作为 2-rank 策略，不能作为通用多机
多卡默认策略。4/8 卡应优先使用真正的 compressed reduce-scatter、分层
collective 或 ReducedShard consumer，避免每个 rank 收集所有量化 payload。

## 测试口径

- 源码提交：`754b8b46a31c41020058d33c7ae1d27f4dfabcda`
- 镜像：`ccdl-comm-a6000:cu126-torch25`
- GPU：NVIDIA RTX A6000，单机使用 2/4 卡；双机每机使用 1/2/4 卡
- 模型：FP16 synthetic MLP，`62,914,560` 参数
- 输入/网络：`2048 -> 4096 x 4 -> 1024`
- batch size：每 rank 16
- 优化器：SGD
- 每次 20 步，前 5 步预热不计时
- 每个配置独立运行 3 次，表格使用中位数
- baseline：PyTorch DDP FP16 gradient all-reduce
- CCDL：INT8、group size 64、all-gather、异步 gather、异步 error-feedback
- 指标：全局 samples/s；保持每 rank batch 不变，属于弱扩展测试

## 性能结果

| 拓扑 | 模式 | 中位 step ms | 中位 samples/s | CCDL/基线 | step 范围 ms | 峰值显存 MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 单机 2 卡 | DDP | 20.102 | 1591.92 | 1.00x | 20.068–20.263 | 376.56 |
| 单机 2 卡 | CCDL INT8 | 16.474 | 1942.46 | **1.22x** | 16.428–16.767 | 1046.44 |
| 单机 4 卡 | DDP | 28.585 | 2238.94 | 1.00x | 28.543–28.700 | 376.56 |
| 单机 4 卡 | CCDL INT8 | 34.799 | 1839.14 | **0.82x** | 34.633–35.346 | 1294.44 |
| 双机 2 卡 | DDP | 1080.496 | 29.62 | 1.00x | 1080.373–1080.591 | 376.56 |
| 双机 2 卡 | CCDL INT8 | 563.250 | 56.81 | **1.92x** | 563.220–563.548 | 1046.44 |
| 双机 4 卡 | DDP | 1618.775 | 39.54 | 1.00x | 1617.692–1620.385 | 376.56 |
| 双机 4 卡 | CCDL INT8 | 1668.504 | 38.36 | **0.97x** | 1668.473–1668.575 | 1294.44 |
| 双机 8 卡 | DDP | 1084.083 | 118.07 | 1.00x | 1083.296–1084.459 | 376.56 |
| 双机 8 卡 | CCDL INT8 | 3886.840 | 32.93 | **0.28x** | 3886.671–3887.069 | 1790.44 |

三次重复的波动很小，说明 4/8 卡变慢不是偶然抖动。

## 数值与训练状态

所有 30 次正式训练均完成，未出现 CUDA、NCCL、非有限 loss 或 rank 状态
错误。在相同拓扑和随机种子下，CCDL 与 DDP 的 20 步平均 train loss 在当前
JSON 精度下相同，记录的 loss delta 均为 `0.0`。

这只能证明短程训练数值 smoke 通过，不代表真实数据集上的最终精度或完整
收敛步数已经得到证明。全量精度结论仍需固定数据顺序、训练 epoch 和评估集
进行长程训练。

## 双机网络限制

两机 `eno2` 实测均为 `1000Mb/s`。两端各有 100Gb/s Mellanox HCA，物理端口
均显示 Active/LinkUp，但 IPoIB 地址配置后不可互相 ping，且两端 SM/LID 所属
fabric 不一致，因此本轮没有伪造 RDMA 可用状态，而是明确设置：

```bash
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=eno2
GLOO_SOCKET_IFNAME=eno2
```

双机绝对吞吐只代表当前 1GbE 环境；CCDL/DDP 同机同口径相对比仍有效。

## 测试中发现并闭环的问题

1. 145 节点使用 Docker 19.03/runc，旧默认 seccomp 会拒绝新镜像的线程
   系统调用，表现为 `pthread_create: EPERM` 和 CUDA `error 304`。加入
   `--security-opt seccomp=unconfined` 后线程、CUDA 和扩展 smoke 全部通过。
   长期建议升级 Docker/containerd，而不是永久依赖该兼容参数。
2. 首次双机 smoke 在 145 worker 上传前启动，导致 156 等待 rendezvous。
   专用容器被定向停止，编排脚本增加远端目录和部署顺序门禁后重跑通过；
   正式数据不包含该失败运行。
3. IPoIB 可达性验证失败后，临时地址和接口状态均恢复到测试前状态；没有将
   不可用 RDMA 路径纳入性能结论。

结构化汇总见 [`summary.json`](summary.json)，每次训练的原始输出见
[`raw/`](raw/)。
