# CCDL 低精度通信库


需要 CUDA 编译环境（nvcc）。
安装

```bash
python csrc/quantization/gen_code_quant.py --output csrc/quantization/
python csrc/quantization/gen_code_dequant.py --output csrc/quantization/
pip install -e .

可以在 `tests/comm` 下找到测试用例, 例如测试 `all_gather`.

```bash
torchrun --nnodes 1 --nproc-per-node 4 ./tests/comm/test_all_gather.py --size 4m
```

## 快速使用

```python
import ccdl.comm as comm
from ccdl.quantization import Quantizer
quantizer = Quantizer(group_size=128, dim=-1, bit=8, topk=0, stochastic=False, dtype=torch.float16)

output_list = [
    torch.empty_like(data) for _ in range(world_size)
]

comm.qall_gather(output_list, rank_data, quantizer=quantizer)
```