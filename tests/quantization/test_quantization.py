import torch
from ccdl.quantization import Quantizer

group_size_list = [16, 32, 64]
bit_list = [4, 8]
stochastic_list = [True, False]
data_shape_list = [(4096,), (4096, 4096), (1024,), (64,)]
topk_list = [0, 1, 2]
dtype_list = ["fp16", "bf16", "fp32"]
quant_type_list = ["linear", "normal", "uniform", "e3m0", "e2m1"]

group_size = 64
topk = 1
bit = 4
stochastic = True
qtz = Quantizer(group_size, -1, bit, topk, stochastic, "fp16")
data = torch.randn((4096,), device='cuda', dtype=torch.half)
d = qtz.quantize(data)
STR_TO_TORCH_DTYPE = {
    "fp16": torch.half,
    "bf16": torch.bfloat16,
    "fp32": torch.float32
}

loop = 100

for group_size in group_size_list:
    for bit in bit_list:
        for topk in topk_list:
            for data_shape in data_shape_list:
                for dtype in dtype_list:
                    for quant_type in quant_type_list:
                        if quant_type != "linear" and bit > 4:
                            continue
                        print(group_size, bit, topk, data_shape, dtype, quant_type)
                        qtz = Quantizer(group_size, -1, bit, topk, False, dtype, quant_type=quant_type)
                        data = torch.randn(data_shape, device='cuda', dtype=STR_TO_TORCH_DTYPE[dtype])
                        q = qtz.quantize(data)
                        dq = qtz.dequantize(q, data_shape)

                        sqtz = Quantizer(group_size, -1, bit, topk, True, dtype, quant_type=quant_type)
                        tmp = torch.zeros_like(data).float()
                        for i in range(loop):
                            sq = sqtz.quantize(data)
                            sdq = sqtz.dequantize(sq, data_shape)
                            tmp += sdq.float()
                        tmp /= loop

                        print((data - dq).abs().max().item(), (data - dq).abs().mean().item(), ((data - dq).abs() / (data.abs() + 1e-6)).mean().item())
                        print((data - tmp).abs().max().item(), (data - tmp).abs().mean().item(), ((data - tmp).abs() / (data.abs() + 1e-6)).mean().item())
