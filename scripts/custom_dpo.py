#!/usr/bin/env python
# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

"""
# Full training
python scripts/custom_dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns

# Teacher prompt:
python scripts/custom_dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --teacher_prompt_instruction "Think step by step and provide your own answer." \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-CustomDPO \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16

# LoRA:
python scripts/custom_dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
"""

import logging
from copy import deepcopy
import os
import sys

import datasets
import torch
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from alignment import DPOConfig, ScriptArguments, TeacherPromptAlignedDPOTrainer, get_dataset, get_model, get_tokenizer
from trl import ModelConfig, TrlParser, get_peft_config
from trl.data_utils import extract_prompt, is_conversational


logger = logging.getLogger(__name__)


def inject_instruction_into_prompt(prompt, instruction):
    instruction = instruction.strip()

    if isinstance(prompt, str):
        return f"{instruction}\n\n{prompt}"

    if is_conversational({"prompt": prompt}):
        teacher_prompt = deepcopy(prompt)
        if teacher_prompt and teacher_prompt[0].get("role") == "system":
            system_message = deepcopy(teacher_prompt[0])
            prefix = system_message.get("content", "")
            system_message["content"] = f"{prefix}\n\n{instruction}".strip() if prefix else instruction
            teacher_prompt[0] = system_message
        else:
            teacher_prompt = [{"role": "system", "content": instruction}] + teacher_prompt
        return teacher_prompt

    raise TypeError(f"Unsupported prompt type for teacher prompt injection: {type(prompt)}")


def build_teacher_prompts(example, shared_instruction=None, chosen_instruction=None, rejected_instruction=None):
    prompt = example["prompt"]
    shared_instruction = shared_instruction.strip() if shared_instruction else None
    chosen_instruction = chosen_instruction.strip() if chosen_instruction else None
    rejected_instruction = rejected_instruction.strip() if rejected_instruction else None

    chosen_effective = chosen_instruction or shared_instruction
    rejected_effective = rejected_instruction or shared_instruction

    if chosen_effective is None and rejected_effective is None:
        return example

    def get_fallback_prompt():
        if "teacher_prompt" in example and example["teacher_prompt"] is not None:
            return deepcopy(example["teacher_prompt"])
        return deepcopy(prompt)

    example["teacher_chosen_prompt"] = (
        inject_instruction_into_prompt(prompt, chosen_effective) if chosen_effective else get_fallback_prompt()
    )
    example["teacher_rejected_prompt"] = (
        inject_instruction_into_prompt(prompt, rejected_effective) if rejected_effective else get_fallback_prompt()
    )

    if chosen_effective is not None and chosen_effective == rejected_effective:
        example["teacher_prompt"] = deepcopy(example["teacher_chosen_prompt"])

    return example


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###################
    # Model & Tokenizer
    ###################
    model = get_model(model_args, training_args)
    ref_model = get_model(model_args, training_args)
    tokenizer = get_tokenizer(model_args, training_args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    #########
    # Dataset
    #########
    dataset = get_dataset(script_args)
    for split in dataset:
        if "prompt" not in dataset[split].column_names:
            dataset[split] = dataset[split].map(extract_prompt, desc=f"Extracting prompt for {split} split")

        if any(
            [
                script_args.teacher_prompt_instruction,
                script_args.teacher_chosen_prompt_instruction,
                script_args.teacher_rejected_prompt_instruction,
            ]
        ):
            dataset[split] = dataset[split].map(
                build_teacher_prompts,
                fn_kwargs={
                    "shared_instruction": script_args.teacher_prompt_instruction,
                    "chosen_instruction": script_args.teacher_chosen_prompt_instruction,
                    "rejected_instruction": script_args.teacher_rejected_prompt_instruction,
                },
                desc=f"Adding teacher prompts for {split} split",
            )

        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    ##########
    # Training
    ##########
    trainer = TeacherPromptAlignedDPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
