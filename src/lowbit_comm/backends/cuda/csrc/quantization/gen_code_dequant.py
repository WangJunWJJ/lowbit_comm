import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir-path", type=str, default="csrc/quantization/")
args = parser.parse_args()
assert os.path.isdir(args.output_dir_path), f"output-dir-path: {args.output_dir_path} is not a valid directory"
file_path = os.path.join(args.output_dir_path, "gen_dequant_api.cu")


headers = """#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "dequant_api.cuh"
#include "dequant_kernel.cuh"
#include "dequant_kernel_fp32.cuh"
#include "enum.cuh"

#define QUANT_SHAREDMEM_SIZE(group_size, threads_per_group) (2*((128 / threads_per_group) * group_size))
"""

dtype_list = ["__half", "__nv_bfloat16", "float"]
torch_dtype_list = ["torch::kHalf", "torch::kBFloat16", "torch::kFloat32"]
group_size_list = ["16", "32", "64"]
topk_list = ["0", "1", "2"]
threads_per_group_list = ["1"]
bit_list = ["4", "8"]
quant_type_list = ["Linear", "Normal", "Uniform", "E2M1", "E3M0"]

template_gen_code = """
template __global__ void dequant_kernel<{dtype}, {group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::NONE, QuantType::{quant_type}, true>(uint16_t* input, uint16_t* output, int64_t length);
template __global__ void dequant_kernel<{dtype}, {group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::NONE, QuantType::{quant_type}, false>(uint16_t* input, uint16_t* output, int64_t length);
template __global__ void dequant_kernel<{dtype}, {group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::SUM, QuantType::{quant_type}, true>(uint16_t* input, uint16_t* output, int64_t length);
template __global__ void dequant_kernel<{dtype}, {group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::SUM, QuantType::{quant_type}, false>(uint16_t* input, uint16_t* output, int64_t length);
template __global__ void dequant_kernel_compact<{dtype}, {group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::NONE, QuantType::{quant_type}, true>(uint16_t* input, uint16_t* output, int64_t length);
template __global__ void dequant_kernel_compact<{dtype}, {group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::NONE, QuantType::{quant_type}, false>(uint16_t* input, uint16_t* output, int64_t length);
template __global__ void dequant_kernel_compact<{dtype}, {group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::SUM, QuantType::{quant_type}, true>(uint16_t* input, uint16_t* output, int64_t length);
template __global__ void dequant_kernel_compact<{dtype}, {group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::SUM, QuantType::{quant_type}, false>(uint16_t* input, uint16_t* output, int64_t length);

void {func_name}(torch::Tensor input, torch::Tensor output, ReduceOP reduce_op = ReduceOP::NONE, bool compact = false) {{
    const int64_t input_bytes_per_group = {input_bytes_per_group};
    TORCH_CHECK(input.numel() % input_bytes_per_group == 0, "input numel should be multiple of input_bytes_per_group");
    c10::cuda::CUDAGuard device_guard(input.device());
    int64_t num_groups = input.numel() / input_bytes_per_group;
    int64_t num_blocks = (num_groups + 127) / 128;
    cudaStream_t stream = get_current_cuda_stream();
    if (num_groups % 128 == 0) {{
        if (reduce_op == ReduceOP::NONE)
            if (!compact)
                dequant_kernel<{dtype}, {group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::NONE, QuantType::{quant_type},true> <<<num_blocks, 128, QUANT_SHAREDMEM_SIZE({group_size}, {threads_per_group}), stream>>> (
                    (uint16_t*)input.data_ptr(), 
                    (uint16_t*)output.data_ptr(),
                    0
                );
            else
                dequant_kernel_compact<{dtype}, {group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::NONE, QuantType::{quant_type},true> <<<num_blocks, 128, QUANT_SHAREDMEM_SIZE({group_size}, {threads_per_group}), stream>>> (
                    (uint16_t*)input.data_ptr(), 
                    (uint16_t*)output.data_ptr(),
                    0
                );
        else if (reduce_op == ReduceOP::SUM) 
            if (!compact)
                dequant_kernel<{dtype}, {group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::SUM, QuantType::{quant_type}, true> <<<num_blocks, 128, QUANT_SHAREDMEM_SIZE({group_size}, {threads_per_group}), stream>>> (
                    (uint16_t*)input.data_ptr(), 
                    (uint16_t*)output.data_ptr(),
                    0
                );
            else
                dequant_kernel_compact<{dtype}, {group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::SUM, QuantType::{quant_type}, true> <<<num_blocks, 128, QUANT_SHAREDMEM_SIZE({group_size}, {threads_per_group}), stream>>> (
                    (uint16_t*)input.data_ptr(), 
                    (uint16_t*)output.data_ptr(),
                    0
                );
    }} else {{
        if (reduce_op == ReduceOP::NONE)
            if (!compact)
                dequant_kernel<{dtype}, {group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::NONE, QuantType::{quant_type}, false> <<<num_blocks, 128, QUANT_SHAREDMEM_SIZE({group_size}, {threads_per_group}), stream>>> (
                    (uint16_t*)input.data_ptr(), 
                    (uint16_t*)output.data_ptr(),
                    num_groups * {group_size}
                );
            else
                dequant_kernel_compact<{dtype}, {group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::NONE, QuantType::{quant_type}, false> <<<num_blocks, 128, QUANT_SHAREDMEM_SIZE({group_size}, {threads_per_group}), stream>>> (
                    (uint16_t*)input.data_ptr(), 
                    (uint16_t*)output.data_ptr(),
                    num_groups * {group_size}
                );
        else if (reduce_op == ReduceOP::SUM) 
            if (!compact)
                dequant_kernel<{dtype}, {group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::SUM, QuantType::{quant_type}, false> <<<num_blocks, 128, QUANT_SHAREDMEM_SIZE({group_size}, {threads_per_group}), stream>>> (
                    (uint16_t*)input.data_ptr(), 
                    (uint16_t*)output.data_ptr(),
                    num_groups * {group_size}
                );
            else
                dequant_kernel_compact<{dtype}, {group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::SUM, QuantType::{quant_type}, false> <<<num_blocks, 128, QUANT_SHAREDMEM_SIZE({group_size}, {threads_per_group}), stream>>> (
                    (uint16_t*)input.data_ptr(), 
                    (uint16_t*)output.data_ptr(),
                    num_groups * {group_size}
                );

    }}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}}


"""

template_gen_code_fp32 = """
template __global__ void dequant_kernel_fp32<{group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::NONE, QuantType::{quant_type}, true>(uint32_t* input, uint32_t* output, int64_t length);
template __global__ void dequant_kernel_fp32<{group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::NONE, QuantType::{quant_type}, false>(uint32_t* input, uint32_t* output, int64_t length);
template __global__ void dequant_kernel_fp32<{group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::SUM, QuantType::{quant_type}, true>(uint32_t* input, uint32_t* output, int64_t length);
template __global__ void dequant_kernel_fp32<{group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::SUM, QuantType::{quant_type}, false>(uint32_t* input, uint32_t* output, int64_t length);
template __global__ void dequant_kernel_fp32_compact<{group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::NONE, QuantType::{quant_type}, true>(uint32_t* input, uint32_t* output, int64_t length);
template __global__ void dequant_kernel_fp32_compact<{group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::NONE, QuantType::{quant_type}, false>(uint32_t* input, uint32_t* output, int64_t length);
template __global__ void dequant_kernel_fp32_compact<{group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::SUM, QuantType::{quant_type}, true>(uint32_t* input, uint32_t* output, int64_t length);
template __global__ void dequant_kernel_fp32_compact<{group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::SUM, QuantType::{quant_type}, false>(uint32_t* input, uint32_t* output, int64_t length);
void {func_name}(torch::Tensor input, torch::Tensor output, ReduceOP reduce_op = ReduceOP::NONE, bool compact=false) {{
    const int64_t input_bytes_per_group = {input_bytes_per_group};
    TORCH_CHECK(input.numel() % input_bytes_per_group == 0, "input numel should be multiple of input_bytes_per_group");
    c10::cuda::CUDAGuard device_guard(input.device());
    int64_t num_groups = input.numel() / input_bytes_per_group;
    int64_t num_blocks = (num_groups + 127) / 128;
    cudaStream_t stream = get_current_cuda_stream();
    if (num_groups % 128 == 0) {{
        if (reduce_op == ReduceOP::NONE)
            if (!compact)
                dequant_kernel_fp32<{group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::NONE, QuantType::{quant_type},true> <<<num_blocks, 128, 2 * QUANT_SHAREDMEM_SIZE({group_size}, {threads_per_group}), stream>>> (
                    (uint32_t*)input.data_ptr(), 
                    (uint32_t*)output.data_ptr(),
                    0
                );
            else
                dequant_kernel_fp32_compact<{group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::NONE, QuantType::{quant_type},true> <<<num_blocks, 128, 2 * QUANT_SHAREDMEM_SIZE({group_size}, {threads_per_group}), stream>>> (
                    (uint32_t*)input.data_ptr(), 
                    (uint32_t*)output.data_ptr(),
                    0
                );
        else if (reduce_op == ReduceOP::SUM) 
            if (!compact)
                dequant_kernel_fp32<{group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::SUM, QuantType::{quant_type}, true> <<<num_blocks, 128, 2 * QUANT_SHAREDMEM_SIZE({group_size}, {threads_per_group}), stream>>> (
                    (uint32_t*)input.data_ptr(), 
                    (uint32_t*)output.data_ptr(),
                    0
                );
            else
                dequant_kernel_fp32_compact<{group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::SUM, QuantType::{quant_type}, true> <<<num_blocks, 128, 2 * QUANT_SHAREDMEM_SIZE({group_size}, {threads_per_group}), stream>>> (
                    (uint32_t*)input.data_ptr(), 
                    (uint32_t*)output.data_ptr(),
                    0
                );
    }} else {{
        if (reduce_op == ReduceOP::NONE)
            if (!compact)
                dequant_kernel_fp32<{group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::NONE, QuantType::{quant_type}, false> <<<num_blocks, 128, 2 * QUANT_SHAREDMEM_SIZE({group_size}, {threads_per_group}), stream>>> (
                    (uint32_t*)input.data_ptr(), 
                    (uint32_t*)output.data_ptr(),
                    num_groups * {group_size}
                );
            else
                dequant_kernel_fp32_compact<{group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::NONE, QuantType::{quant_type}, false> <<<num_blocks, 128, 2 * QUANT_SHAREDMEM_SIZE({group_size}, {threads_per_group}), stream>>> (
                    (uint32_t*)input.data_ptr(), 
                    (uint32_t*)output.data_ptr(),
                    num_groups * {group_size}
                );
        else if (reduce_op == ReduceOP::SUM) 
            if (!compact)
                dequant_kernel_fp32<{group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::SUM, QuantType::{quant_type}, false> <<<num_blocks, 128, 2 * QUANT_SHAREDMEM_SIZE({group_size}, {threads_per_group}), stream>>> (
                    (uint32_t*)input.data_ptr(), 
                    (uint32_t*)output.data_ptr(),
                    num_groups * {group_size}
                );
            else
                dequant_kernel_fp32_compact<{group_size}, {topk}, {threads_per_group}, {bit}, ReduceOP::SUM, QuantType::{quant_type}, false> <<<num_blocks, 128, 2 * QUANT_SHAREDMEM_SIZE({group_size}, {threads_per_group}), stream>>> (
                    (uint32_t*)input.data_ptr(), 
                    (uint32_t*)output.data_ptr(),
                    num_groups * {group_size}
                );
    }}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}}


"""

txt = headers

def get_func_name(dtype, group_size, topk, threads_per_group, bit, quant_type):
    def dtype_to_str(dtype):
        if dtype == "__half":
            return "fp16"
        elif dtype == "__nv_bfloat16":
            return "bf16"
        elif dtype == "float":
            return "fp32"

    return f"dequant_{dtype_to_str(dtype)}_g{group_size}_top{topk}_{bit}bit_{quant_type}"

inplace_func = """
void inplace_dequantize(torch::Tensor input, torch::Tensor output, int64_t group_size, int64_t topk, int64_t bit, ReduceOP reduce_op, QuantType quant_type, bool compact) {
    TORCH_CHECK(bit == 8 || bit == 4, "bit must be 8 or 4");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "input must be uint8");
""" + '\n'

def get_if(cond: str, _i, indent):
    if _i == 0:
        return indent * " " + f"if ({cond}) {{\n"
    else:
        return indent * " " + f"}} else if ({cond}) {{\n"

inner_template = [
    "{func_name}(input, output, reduce_op, compact);",
    "return;",
]

def gen_inner(dtype, group_size, bit, topk, indent, quant_type):
    
    func_name = get_func_name(dtype, group_size, topk, "1", bit, quant_type)
    s_list = [line.format(func_name=func_name) for line in inner_template]
    s = ""
    for line in s_list:
        s += indent * " " + line + "\n"
    global txt
    template = None
    if dtype == "float":
        input_bytes_per_group = (int(group_size) * int(bit) // 8) + 4
        if int(topk) == 1:
            input_bytes_per_group += 8
        elif int(topk) == 2:
            input_bytes_per_group += 12
        template = template_gen_code_fp32
    else:
        input_bytes_per_group = (int(group_size) * int(bit) // 8) + 2
        if int(topk) == 1:
            input_bytes_per_group += 4
        elif int(topk) == 2:
            input_bytes_per_group += 6

        template = template_gen_code

    txt += template.format(
        dtype=dtype, 
        group_size=group_size, 
        topk=topk, 
        threads_per_group="1", 
        bit=bit, 
        func_name=func_name,
        input_bytes_per_group=input_bytes_per_group,
        quant_type = quant_type
    )
    return s

indent = 4

for _i, (torch_dtype, dtype) in enumerate(zip(torch_dtype_list, dtype_list)):
    inplace_func += get_if(f"output.dtype() == {torch_dtype}", _i, indent)
    indent += 4

    for _j, bit in enumerate(bit_list):
        inplace_func += get_if(f"bit == {bit}", _j, indent)
        indent += 4
        for _k, group_size in enumerate(group_size_list):
            inplace_func += get_if(f"group_size == {group_size}", _k, indent)
            indent += 4
            for _l, topk in enumerate(topk_list):
                inplace_func += get_if(f"topk == {topk}", _l, indent)
                indent += 4

                for _m, quant_type in enumerate(quant_type_list):
                    if quant_type != "Linear" and bit != "4":
                        continue
                    inplace_func += get_if(f"quant_type == QuantType::{quant_type}", _m, indent)
                    inplace_func += gen_inner(dtype, group_size, bit, topk, indent + 4, quant_type)

                inplace_func += indent * " " + "}\n"
                indent -= 4

            inplace_func += indent * " " + "}\n"
            indent -= 4

        inplace_func += indent * " " + "}\n"
        indent -= 4

    inplace_func += indent * " " + "}\n"
    indent -= 4



inplace_func += indent * " " + "}\n"
inplace_func += indent * " " + "assert(false);\n}\n"

txt += inplace_func

func = """

torch::Tensor dequantize(torch::Tensor input, int64_t group_size, int64_t topk, int64_t bit, ReduceOP reduce_op, QuantType quant_type, DType dtype, bool compact) {
    TORCH_CHECK(bit == 8 || bit == 4, "bit must be 8 or 4");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "input must be uint8");
    auto torch_dtype = torch::kHalf;
    if (dtype == DType::FP16)
        torch_dtype = torch::kHalf;
    else if (dtype == DType::BF16)
        torch_dtype = torch::kBFloat16;
    else if (dtype == DType::FP32)
        torch_dtype = torch::kFloat32;

    auto options = torch::TensorOptions().dtype(torch_dtype).device(input.device());
    int64_t input_bytes_per_group;
    if (dtype == DType::FP32) {
        input_bytes_per_group = group_size * bit / 8 + 4;
        if (topk == 1) input_bytes_per_group += 8;
        else if (topk == 2) input_bytes_per_group += 12;
    } else {
        input_bytes_per_group = group_size * bit / 8 + 2;
        if (topk == 1) input_bytes_per_group += 4;
        else if (topk == 2) input_bytes_per_group += 6;
    }
    int64_t num_groups = input.numel() / input_bytes_per_group;
    torch::Tensor output = torch::empty({num_groups * group_size}, options);
    inplace_dequantize(input, output, group_size, topk, bit, reduce_op, quant_type, compact);
    return output;
}
"""

txt += func

f = open(file_path, "w+")
f.write(txt)
f.close()
