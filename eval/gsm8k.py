"""Evaluate a base model or a QeRL checkpoint on GSM8K / AMC23 with vLLM.

Examples:
    # NVFP4 (or 16-bit) base model
    python -m eval.gsm8k --model /path/to/Qwen2.5-7B-Instruct-NVFP4A16 --data gsm8k
    # QeRL checkpoint (LoRA + RMSNorm weights) on top of the base model recorded in its adapter_config.json.
    # The adapter is applied unmerged, so an NVFP4 base keeps running on packed 4-bit weights.
    python -m eval.gsm8k --model ckpt/run/checkpoint-900 --data gsm8k

For GSM8K two accuracies are reported:
    strict  - the text inside <answer></answer> is exactly the gold answer (the criterion used as training reward)
    numeric - the last number inside <answer></answer> (or in the whole response when the tags are missing)
              equals the gold answer
"""

import argparse
import json
import os
import re
from dataclasses import asdict

import torch
from datasets import Dataset, load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from peft import PeftConfig
from safetensors.torch import load_file as safe_load_file
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from trl_trainer.runtime_patches import apply_patches, make_tensor_lora_request_cls

SYSTEM_PROMPT = """
        Solve the following math problem step by step. The reasoning process and direct answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>: 
        <think>
        ...
        </think>
        <answer>
        ...
        </answer>
    """


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def get_gsm8k_questions() -> Dataset:
    data = load_dataset("openai/gsm8k", "main", split="test")
    return data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )


def get_amc_questions() -> Dataset:
    data = load_dataset("math-ai/amc23", split="test")
    return data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": x["answer"],
        }
    )


def load_model(model_name_or_path: str, gpu_memory_utilization: float, max_model_len: int):
    """Returns (llm, tokenizer, lora_request). `lora_request` is None for a plain model."""
    apply_patches()  # also makes Triton usable on GPUs newer than its bundled ptxas
    adapter_config = os.path.join(model_name_or_path, "adapter_config.json")
    if not os.path.isfile(adapter_config):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        llm = LLM(model=model_name_or_path, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len)
        return llm, tokenizer, None

    # The adapter weights are pushed into the engine in-process, which requires it to live in this process.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    peft_config = PeftConfig.from_pretrained(model_name_or_path)
    base_path = peft_config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(base_path)

    adapter_path = os.path.join(model_name_or_path, "adapter_model.safetensors")
    if os.path.exists(adapter_path):
        adapter_weights = safe_load_file(adapter_path)
    else:
        adapter_weights = torch.load(os.path.join(model_name_or_path, "adapter_model.bin"), map_location="cpu")

    # LoRA matrices go into a vLLM adapter; fully trained modules (`modules_to_save`, e.g. the RMSNorm weights)
    # replace the corresponding base weights.
    lora_weights, full_weights = {}, {}
    for name, tensor in adapter_weights.items():
        if "lora_" in name:
            lora_weights[name] = tensor
        else:
            name = name.removeprefix("base_model.model.").replace("modules_to_save.default.", "")
            full_weights[name] = tensor
    print(f"Adapter: {len(lora_weights)} LoRA tensors, {len(full_weights)} fully trained tensors (base: {base_path})")

    llm = LLM(
        model=base_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_lora=bool(lora_weights),
        max_lora_rank=peft_config.r if lora_weights else None,
    )
    if full_weights:
        llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights(list(full_weights.items()))

    lora_request = None
    if lora_weights:
        config = asdict(peft_config)
        config["modules_to_save"] = None  # handled above; vLLM rejects adapters declaring it
        lora_request = make_tensor_lora_request_cls()(
            lora_name="qerl-eval",
            lora_int_id=1,
            lora_path="in-memory",
            peft_config=config,
            lora_tensors=lora_weights,
        )
        llm.llm_engine.add_lora(lora_request)
    return llm, tokenizer, lora_request


_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


def _to_number(text: str):
    matches = _NUMBER.findall(text.replace("$", ""))
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def score_gsm8k(response: str, gold: str) -> dict:
    extracted = extract_xml_answer(response)
    gold_number = _to_number(gold)
    predicted_number = _to_number(extracted)
    return {
        "strict": float(extracted == gold),
        "numeric": float(
            gold_number is not None and predicted_number is not None and abs(predicted_number - gold_number) < 1e-6
        ),
    }


def score_math_verify(response: str, gold: str) -> dict:
    """Symbolic equivalence of the (boxed / tagged) answer with the gold answer."""
    if extract_xml_answer(response) == gold:
        return {"accuracy": 1.0}
    gold_parsed = parse(gold, extraction_mode="first_match")
    if len(gold_parsed) == 0:
        return {"accuracy": 0.0}
    answer_parsed = parse(
        response,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    equations=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
    )
    try:
        return {"accuracy": float(verify(gold_parsed, answer_parsed))}
    except Exception:
        return {"accuracy": 0.0}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, required=True, help="Model, or QeRL checkpoint directory.")
    parser.add_argument("--data", type=str, default="gsm8k", choices=["gsm8k", "amc"])
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy decoding.")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N problems.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load", type=str, default=None, help="Deprecated and ignored: the model type is detected.")
    return parser.parse_args()


def main():
    args = parse_args()
    data = get_gsm8k_questions() if args.data == "gsm8k" else get_amc_questions()
    if args.limit:
        data = data.select(range(min(args.limit, len(data))))
    score = score_gsm8k if args.data == "gsm8k" else score_math_verify

    llm, tokenizer, lora_request = load_model(args.model, args.gpu_memory_utilization, args.max_model_len)

    prompts = [
        tokenizer.apply_chat_template(sample["prompt"], tokenize=False, add_generation_prompt=True) for sample in data
    ]
    sampling_params = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed
    )
    # One batched call: vLLM schedules all problems concurrently.
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=[lora_request] * len(prompts) if lora_request is not None else None,
    )
    responses = [output.outputs[0].text for output in outputs]

    records = []
    for sample, response in zip(data, responses):
        records.append(
            {
                "question": sample["prompt"][-1]["content"],
                "gold": sample["answer"],
                "response": response,
                **score(response, sample["answer"]),
            }
        )
    metric_names = [key for key in records[0] if key not in ("question", "gold", "response")]
    summary = {name: sum(record[name] for record in records) / len(records) for name in metric_names}
    summary.update(model=args.model, data=args.data, num_questions=len(records), temperature=args.temperature)
    print(json.dumps(summary, indent=2))

    os.makedirs(args.output_dir, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model.strip("/"))[-120:]
    with open(os.path.join(args.output_dir, f"{tag}_{args.data}_results.json"), "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=1)


if __name__ == "__main__":
    main()
