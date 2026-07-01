# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import os

os.environ["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = "36000"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    # Create an LLM.
    llm = LLM(
        model="/data/chengjie/models/Qwen/Qwen3.5-35B-A3B",
        enforce_eager=True,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        tensor_parallel_size=4,
        async_scheduling=False,
        enable_expert_parallel=True,
        enable_eplb=True,
        eplb_config={
            "window_size":  1,
            "step_interval": 1, 
            "num_redundant_experts": 4,
            "log_balancedness": True,
            # "use_async": True,  # 需要非阻塞 EPLB 时再开
        },
    )
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
