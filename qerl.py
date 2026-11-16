# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import os
from utils.rewards import (
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
        dapo_score,
        code_reward,
        accuracy_reward,
        think_reward_func,
        gsm8k_score
    )
from utils.data import get_gsm8k_questions, get_dapo_questions,get_code_questions,get_bigmath_questions,get_mm_questions
from utils.configs import DataConfig, TrainingConfig, QeRLConfig, ModelConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_flash_attn_2_available
from trl_trainer.grpo_trainer import GRPOTrainer
from trl_trainer.runtime_patches import unwrap_compressed_linears
from peft import LoraConfig, get_peft_model
from trl import TrlParser

def get_dataset(data):
    if data == "gsm8k":
        dataset = get_gsm8k_questions(split="train")
        reward_funcs = [
            soft_format_reward_func,
            correctness_reward_func,
        ]
    elif data == "bigmath" or data == "bigmath_hard":
        dataset = get_bigmath_questions(data, split="train")
        reward_funcs = [
            soft_format_reward_func,
            accuracy_reward,
        ]
    elif data == "dapomath":
        dataset = get_dapo_questions(split="train")
        reward_funcs = [dapo_score]
    elif data == "miromind":
        dataset = get_mm_questions(split="train")
        reward_funcs = [dapo_score]
    elif data == "codeforces":
        dataset = get_code_questions()
        reward_funcs = [code_reward]
    else:
        raise ValueError(f"Unsupported dataset: {data}")
    return dataset, reward_funcs

def main(data_args: DataConfig, training_args: QeRLConfig, model_args: ModelConfig):

    data = data_args.dataset
    dataset, reward_funcs = get_dataset(data)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name,
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "sdpa",
        dtype='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_args.model_name)

    # NVFP4 layers are dequantized on the fly in the forward pass (see trl_trainer/runtime_patches.py)
    unwrap_compressed_linears(model)
    # the adaptive quantization noise is only used on top of a quantized (NVFP4) model
    noise_scheduler = any(hasattr(module, "weight_packed") for module in model.modules())

    # Freeze the base model in its own dtype (prepare_model_for_kbit_training would upcast all 16-bit weights
    # to float32); gradient checkpointing is enabled by the trainer.
    for param in model.parameters():
        param.requires_grad_(False)
    peft_config = LoraConfig(
        r=model_args.lora_rank,  # Rank dimension - typically between 4-32
        lora_alpha=model_args.lora_alpha,  # LoRA scaling factor - typically 2x rank
        target_modules=model_args.target_modules,  # Which modules to apply LoRA to
        modules_to_save=["input_layernorm","post_attention_layernorm"] if model_args.ln else None
    )
    model = get_peft_model(model, peft_config)
    # trainable weights are updated in float32
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    trainer = GRPOTrainer(
        model = model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args = training_args,
        train_dataset = dataset,
        sigma_start = model_args.sigma_start,
        sigma_end = model_args.sigma_end,
        num_stages = model_args.num_stages,
        noise_scheduler = noise_scheduler
    )
    output_dir = training_args.output_dir
    has_checkpoint = os.path.isdir(output_dir) and any(d.startswith("checkpoint-") for d in os.listdir(output_dir))
    trainer.train(resume_from_checkpoint = True if has_checkpoint else None)
    trainer.save_model(os.path.join(output_dir, "final"))
    trainer.save_state()

if __name__ == "__main__":
    parser = TrlParser((DataConfig, QeRLConfig, ModelConfig))
    data_args, training_args, model_args = parser.parse_args_and_config()
    main(data_args, training_args, model_args)
