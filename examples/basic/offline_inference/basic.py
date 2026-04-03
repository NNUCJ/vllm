# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

os.environ["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = "36000"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_TORCH_PROFILER_DIR"] = "/data/chengjie/vllm_project/profiling"

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
] * 2 
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)


def main():
    # Create an LLM.
    llm = LLM(model="/data/models/deepseek/deepseek-moe-16b-base",
            enforce_eager=True,
            gpu_memory_utilization=0.8,
            tensor_parallel_size=8,
            async_scheduling=True,
            trust_remote_code=True,
            profiler_config={"profiler": "torch", "torch_profiler_dir": "/data/chengjie/vllm_project/profiling"})
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    llm.start_profile()
    outputs = llm.generate(prompts, sampling_params)
    llm.stop_profile()
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