import os
import argparse
from torch.cuda import device_count

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--output", type=str)
parser.add_argument("--cuda_device", type=str, default="1")
parser.add_argument("--base_model", type=str, default="unsloth/llama-3-8b-Instruct-bnb-4bit")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--r", type=int, default=64)
parser.add_argument("--alpha", type=int, default=128)
parser.add_argument("--max_seq_length", type=int, default=4096)
parser.add_argument("--resume", type=bool, default=False)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--seed", type=int, default=2448)
parser.add_argument("--scheduler", type=str, default="cosine")
parser.add_argument("--warmup", type=int, default=150)
parser.add_argument("--load_4_bit", type=bool, default=True)

args = parser.parse_args()
args = vars(args)


os.environ["CUDA_VISIBLE_DEVICES"] = args["cuda_device"]


from unsloth import FastLanguageModel, PatchDPOTrainer
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import ORPOConfig, ORPOTrainer


dataset_path = args["data_path"]

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args["base_model"],
    max_seq_length = args["max_seq_length"] ,
    dtype = None, # None is autodetection for unsloth
    load_in_4bit = args["load_4_bit"],
)

model = FastLanguageModel.get_peft_model(
    model,
    r = args['r'],
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = args['alpha'],
    lora_dropout = 0.05,  # Supports any, but = 0 is optimized
    bias = "none",  # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    random_state = args["seed"],
    use_rslora = False,  
    loftq_config = None, 
)

dataset = load_dataset("json", data_files=dataset_path, split="train")
dataset = dataset.shuffle(seed=args["seed"])

PatchDPOTrainer()

trainer = ORPOTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = ORPOConfig(max_length = args["max_seq_length"],
        max_prompt_length = args["max_seq_length"]//2,
        max_completion_length = args["max_seq_length"]//2,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        beta = 0.1,
        logging_steps = 10,
        optim = "adamw_8bit",
        lr_scheduler_type = args["scheduler"],
        num_train_epochs=args["epochs"],
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        output_dir = args["output"],
        seed = args["seed"],
    ),
)

trainer.train(resume_from_checkpoint=args["resume"])

trainer.save_model(args["output"])
model.save_pretrained_merged(f'{args["output"]}_merged', tokenizer, save_method = "merged_16bit",)