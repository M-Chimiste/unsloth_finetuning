import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--output", type=str)
parser.add_argument("--cuda_device", type=str, default="1")
parser.add_argument("--base_model", type=str, default="unsloth/llama-3-8b-bnb-4bit")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--r", type=int, default=128)
parser.add_argument("--alpha", type=int, default=128)
parser.add_argument("--max_seq_length", type=int, default=2048)
parser.add_argument("--resume", type=bool, default=False)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--seed", type=int, default=2448)

args = parser.parse_args()
args = vars(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args["cuda_device"]

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import random

max_seq_length = args["max_seq_length"] 
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


dataset_path = args["data_path"]


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args["base_model"],
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = args['r'], # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = args['alpha'],
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    random_state = args["seed"],
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

dataset = load_dataset("json", data_files=dataset_path, split="train")
dataset = dataset.shuffle(seed=args["seed"])

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "prompt",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        num_train_epochs=args["epochs"],
        save_strategy="steps",
        save_total_limit=3,
        save_steps=200,
        warmup_steps = 150,
        learning_rate = args["lr"],
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = args["seed"],
        output_dir = args["output"],
    ),
)

trainer.train(resume_from_checkpoint=args["resume"])

trainer.save_model(args["output"])
model.save_pretrained_merged(f'{args["output"]}_merged', tokenizer, save_method = "merged_16bit",)
