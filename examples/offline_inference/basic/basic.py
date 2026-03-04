# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import os 
os.environ["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = "36000"
# os.environ["VLLM_TORCH_PROFILER_DIR"] = "/data/chengjie/vllm_project/vllm/profiling/async_nsys"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from vllm import LLM, SamplingParams
# Sample prompts.
# prompts = [
#     "A" * 192
# ] * 8 
# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]
prompts = [
    "以下是中国关于计算机网络的单项选择题，请选出其中的正确答案。\n问题：已知当前TCP连接的RTT值为35ms，连续收到3个确认报文段，它们比相应的数据报文段的发送时间滞后了27ms、30ms与21ms。假设α=0.2，则第三个确认报文段到达后新的RTT估计值为____。\n选项：\nA. 33.4ms\nB. 32.7ms\nC. 21ms\nD. 30.4ms让我们一步一步思考。答案:"
]
# prompts = [
#     "把下面的现代文翻译成文言文：到了春风和煦，阳光明媚的时候，湖面平静，没有惊涛骇浪，天色湖光相连，一片碧绿，广阔无际；沙洲上的鸥鸟，时而飞翔，时而停歇，美丽的鱼游来游去，岸上与小洲上的花草，青翠欲滴。"
# ]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.6, top_k=1, top_p=0.9, max_tokens=5000, presence_penalty=0.05, frequency_penalty=0.02, repetition_penalty=1.05)

# /data/chengjie/models/Qwen/Qwen3-4B
# /data/chengjie/models/Qwen/Qwen3-8B
def main():
    # Create an LLM.
    llm = LLM(model="/data/chengjie/models/Qwen/Qwen3-4B",
              enforce_eager=True,
              max_model_len=8192,
              gpu_memory_utilization=0.8,
              tensor_parallel_size=2,
              async_scheduling=False)
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    # llm.start_profile()
    outputs = llm.generate(prompts, sampling_params)
    # llm.stop_profile()
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
