import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir-path", type=str, default="csrc/quantization/")
args = parser.parse_args()
assert os.path.isdir(args.output_dir_path), f"output-dir-path: {args.output_dir_path} is not a valid directory"
file_path = os.path.join(args.output_dir_path, "gen_quant_api.cu")


headers = """#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <THC/THCAtomics.cuh>
#include <ATen/cuda/CUDAGeneratorImpl.h>

#include "quant_api.cuh"
#include "quant_kernel.cuh"
#include "quant_kernel_fp32.cuh"

#define QUANT_SHAREDMEM_SIZE(group_size, threads_per_group, topk) (2*((128 / threads_per_group) * group_size))
const std::pair<uint64_t, uint64_t> default_rng_engine_inputs = std::make_pair(0, 0);

std::pair<uint64_t, uint64_t> get_rng_seeds(int64_t increment) {
    auto gen = at::check_generator<at::CUDAGeneratorImpl>(at::cuda::detail::getDefaultCUDAGenerator());
    std::pair<uint64_t, uint64_t> rng_engine_inputs;
    {
        std::lock_guard<std::mutex> lock(gen->mutex_);
        rng_engine_inputs = gen->philox_engine_inputs(increment);
    }
    return rng_engine_inputs;
}


"""


torch_dtype_list = ["torch::kHalf", "torch::kBFloat16", "torch::kFloat32"]
dtype_list = ["__half", "__nv_bfloat16", "float"]
group_size_list = ["16", "32", "64"]
topk_list = ["0", "1", "2"]
threads_per_group_list = ["1"]
bit_list = ["4", "8"]
quant_type_list = ["Linear", "Normal", "Uniform", "E2M1", "E3M0"]

template_gen_code = """
template __global__ void quant_kernel<{dtype}, {group_size}, {topk}, {stochastic}, {threads_per_group}, {bit}, QuantType::{quant_type}, true>(uint16_t* input, uint16_t* output, std::pair<uint64_t, uint64_t> seed, int64_t input_len);
template __global__ void quant_kernel<{dtype}, {group_size}, {topk}, {stochastic}, {threads_per_group}, {bit}, QuantType::{quant_type}, false>(uint16_t* input, uint16_t* output, std::pair<uint64_t, uint64_t> seed, int64_t input_len);
template __global__ void quant_kernel_compact<{dtype}, {group_size}, {topk}, {stochastic}, {threads_per_group}, {bit}, QuantType::{quant_type}, true>(uint16_t* input, uint16_t* output, std::pair<uint64_t, uint64_t> seed, int64_t input_len);
template __global__ void quant_kernel_compact<{dtype}, {group_size}, {topk}, {stochastic}, {threads_per_group}, {bit}, QuantType::{quant_type}, false>(uint16_t* input, uint16_t* output, std::pair<uint64_t, uint64_t> seed, int64_t input_len);
void {func_name}(torch::Tensor input, torch::Tensor output, bool compact = false) {{
    TORCH_CHECK(input.numel() % {group_size} == 0, "input numel should be multiple of group_size");
    int64_t num_blocks = (input.numel() + {group_size} * 128 - 1) / ({group_size} * 128);
    cudaStream_t stream = get_current_cuda_stream();
    {get_rng_engine_inputs}
    if (input.numel() % (128 * {group_size}) == 0) {{
        if (!compact)
            quant_kernel<{dtype}, {group_size}, {topk}, {stochastic}, {threads_per_group}, {bit}, QuantType::{quant_type}, true> <<<num_blocks, 128, QUANT_SHAREDMEM_SIZE({group_size}, {threads_per_group}, {topk}), stream>>> (
                (uint16_t*)input.data_ptr(),
                (uint16_t*)output.data_ptr(),
                rng_engine_inputs,
                0
            );
        else
            quant_kernel_compact<{dtype}, {group_size}, {topk}, {stochastic}, {threads_per_group}, {bit}, QuantType::{quant_type}, true> <<<num_blocks, 128, QUANT_SHAREDMEM_SIZE({group_size}, {threads_per_group}, {topk}), stream>>> (
                (uint16_t*)input.data_ptr(),
                (uint16_t*)output.data_ptr(),
                rng_engine_inputs,
                0
            );
    }} else {{
        if (!compact)
            quant_kernel<{dtype}, {group_size}, {topk}, {stochastic}, {threads_per_group}, {bit}, QuantType::{quant_type}, false> <<<num_blocks, 128, QUANT_SHAREDMEM_SIZE({group_size}, {threads_per_group}, {topk}), stream>>> (
                (uint16_t*)input.data_ptr(),
                (uint16_t*)output.data_ptr(),
                rng_engine_inputs,
                input.numel()
            );
        else
            quant_kernel_compact<{dtype}, {group_size}, {topk}, {stochastic}, {threads_per_group}, {bit}, QuantType::{quant_type}, false> <<<num_blocks, 128, QUANT_SHAREDMEM_SIZE({group_size}, {threads_per_group}, {topk}), stream>>> (
                (uint16_t*)input.data_ptr(),
                (uint16_t*)output.data_ptr(),
                rng_engine_inputs,
                input.numel()
            );
    }}
}}


"""

template_gen_code_fp32 = """
template __global__ void quant_kernel_fp32<{group_size}, {topk}, {stochastic}, {threads_per_group}, {bit}, QuantType::{quant_type}, true>(uint32_t* input, uint32_t* output, std::pair<uint64_t, uint64_t> seed, int64_t input_len);
template __global__ void quant_kernel_fp32<{group_size}, {topk}, {stochastic}, {threads_per_group}, {bit}, QuantType::{quant_type}, false>(uint32_t* input, uint32_t* output, std::pair<uint64_t, uint64_t> seed, int64_t input_len);
template __global__ void quant_kernel_fp32_compact<{group_size}, {topk}, {stochastic}, {threads_per_group}, {bit}, QuantType::{quant_type}, true>(uint32_t* input, uint32_t* output, std::pair<uint64_t, uint64_t> seed, int64_t input_len);
template __global__ void quant_kernel_fp32_compact<{group_size}, {topk}, {stochastic}, {threads_per_group}, {bit}, QuantType::{quant_type}, false>(uint32_t* input, uint32_t* output, std::pair<uint64_t, uint64_t> seed, int64_t input_len);
void {func_name}(torch::Tensor input, torch::Tensor output, bool compact = false) {{
    TORCH_CHECK(input.numel() % {group_size} == 0, "input numel should be multiple of group_size");
    int64_t num_blocks = (input.numel() + {group_size} * 128 - 1) / ({group_size} * 128);
    cudaStream_t stream = get_current_cuda_stream();
    {get_rng_engine_inputs}
    if (input.numel() % (128 * {group_size}) == 0) {{
        if (!compact)
            quant_kernel_fp32<{group_size}, {topk}, {stochastic}, {threads_per_group}, {bit}, QuantType::{quant_type}, true> <<<num_blocks, 128, 2 * QUANT_SHAREDMEM_SIZE({group_size}, {threads_per_group}, {topk}), stream>>> (
                (uint32_t*)input.data_ptr(),
                (uint32_t*)output.data_ptr(),
                rng_engine_inputs,
                0
            );
        else
            quant_kernel_fp32_compact<{group_size}, {topk}, {stochastic}, {threads_per_group}, {bit}, QuantType::{quant_type}, true> <<<num_blocks, 128, 2 * QUANT_SHAREDMEM_SIZE({group_size}, {threads_per_group}, {topk}), stream>>> (
                (uint32_t*)input.data_ptr(),
                (uint32_t*)output.data_ptr(),
                rng_engine_inputs,
                0
            );

    }} else {{
        if (!compact)
            quant_kernel_fp32<{group_size}, {topk}, {stochastic}, {threads_per_group}, {bit}, QuantType::{quant_type}, false> <<<num_blocks, 128, 2 * QUANT_SHAREDMEM_SIZE({group_size}, {threads_per_group}, {topk}), stream>>> (
                (uint32_t*)input.data_ptr(),
                (uint32_t*)output.data_ptr(),
                rng_engine_inputs,
                input.numel()
            );
        else 
            quant_kernel_fp32_compact<{group_size}, {topk}, {stochastic}, {threads_per_group}, {bit}, QuantType::{quant_type}, false> <<<num_blocks, 128, 2 * QUANT_SHAREDMEM_SIZE({group_size}, {threads_per_group}, {topk}), stream>>> (
                (uint32_t*)input.data_ptr(),
                (uint32_t*)output.data_ptr(),
                rng_engine_inputs,
                input.numel()
            );
    }}
}}


"""


txt = headers

def get_func_name(dtype, group_size, topk, stochastic, threads_per_group, bit, quant_type):
    def dtype_to_str(dtype):
        if dtype == "__half":
            return "fp16"
        elif dtype == "__nv_bfloat16":
            return "bf16"
        elif dtype == "float":
            return "fp32"

    return f"quant_{dtype_to_str(dtype)}_g{group_size}_top{topk}_{bit}bit_{quant_type}" + ("_stochastic" if stochastic == "true" else "")

def get_rng_engine_inputs(stochastic):
    if (stochastic == "true"):
        return "auto rng_engine_inputs = get_rng_seeds(input.numel());"
    else:
        return "const std::pair<uint64_t, uint64_t> rng_engine_inputs = default_rng_engine_inputs;"

inplace_func = """
void inplace_quantize(torch::Tensor input, torch::Tensor output, int64_t group_size, int64_t topk, bool stochastic, int64_t bit, QuantType quant_type, bool compact) {
    TORCH_CHECK(input.numel() % group_size == 0, "input numel should be multiple of group_size");
    int64_t num_groups = input.numel() / group_size;
    int64_t bytes_per_group;
    if (input.dtype() == torch::kFloat32) {
        bytes_per_group = (group_size * bit) / 8 + 4;
        if (topk == 1) bytes_per_group += 8;
        else if (topk == 2) bytes_per_group += 12;
    } else {
        bytes_per_group = (group_size * bit) / 8 + 2;
        if (topk == 1) bytes_per_group += 4;
        else if (topk == 2) bytes_per_group += 6;
    }
    TORCH_CHECK(output.numel() == num_groups * bytes_per_group, "Output tensor must have the same number of elements as the quantized tensor");
    TORCH_CHECK(output.dtype() == torch::kU8, "Output tensor must be of u8 precision");
""" + '\n'

def get_if(cond: str, _i, indent):
    if _i == 0:
        return indent * " " + f"if ({cond}) {{\n"
    else:
        return indent * " " + f"}} else if ({cond}) {{\n"

inner_template = [
    "if (stochastic) {{",
    "    {func_name_stochastic}(input, output, compact);",
    "    return;",
    "}} else {{",
    "    {func_name}(input, output, compact);",
    "    return;",
    "}}"
]

def gen_inner(dtype, group_size, bit, topk, indent, quant_type):
    func_name = get_func_name(dtype, group_size, topk, "false", "1", bit, quant_type)
    func_name_stochastic = get_func_name(dtype, group_size, topk, "true", "1", bit, quant_type)
    s_list = [line.format(func_name=func_name, func_name_stochastic=func_name_stochastic) for line in inner_template]
    s = ""
    for line in s_list:
        s += indent * " " + line + "\n"
    global txt
    if dtype == "float":
        template = template_gen_code_fp32
    else:
        template = template_gen_code

    txt += template.format(
        dtype=dtype, 
        group_size=group_size, 
        topk=topk, 
        stochastic="false", 
        threads_per_group="1", 
        bit=bit, 
        func_name=func_name,
        get_rng_engine_inputs=get_rng_engine_inputs("false"),
        quant_type=quant_type
    )
    txt += template.format(
        dtype=dtype, 
        group_size=group_size, 
        topk=topk, 
        stochastic="true", 
        threads_per_group="1", 
        bit=bit, 
        func_name=func_name_stochastic,
        get_rng_engine_inputs=get_rng_engine_inputs("true"),
        quant_type=quant_type
    )
    return s

indent = 4
for _i, (torch_dtype, dtype) in enumerate(zip(torch_dtype_list, dtype_list)):
    inplace_func += get_if(f"input.dtype() == {torch_dtype}", _i, indent)
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
torch::Tensor quantize(torch::Tensor input, int64_t group_size, int64_t topk, bool stochastic, int64_t bit, QuantType quant_type, bool compact) {
    auto options = torch::TensorOptions().dtype(torch::kU8).device(input.device());
    int64_t num_groups = (input.numel() + group_size - 1) / group_size;
    int64_t bytes_per_group;
    if (input.dtype() == torch::kFloat32) {
        bytes_per_group = (group_size * bit) / 8 + 4;
        if (topk == 1) bytes_per_group += 8;
        else if (topk == 2) bytes_per_group += 12;
    } else {
        bytes_per_group = (group_size * bit) / 8 + 2;
        if (topk == 1) bytes_per_group += 4;
        else if (topk == 2) bytes_per_group += 6;
    }
    torch::Tensor output = torch::empty({num_groups * bytes_per_group}, options);
    inplace_quantize(input, output, group_size, topk, stochastic, bit, quant_type, compact);
    return output;
}
"""

txt += func

f = open(file_path, "w+")
f.write(txt)
f.close()