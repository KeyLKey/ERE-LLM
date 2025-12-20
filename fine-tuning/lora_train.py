from datasets import Dataset
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
model_name = "/home/keytoliu/L/Qwen2.5-3B-Instruct"
df = pd.read_json('/home/keytoliu/test_3/SCIERC/fine-tuning/data/Step5_Sentence_Have_Type_yes_or_no.json')
ds = Dataset.from_pandas(df)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
data_name = "SCIERC"
def process_func(example):
    MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []

    instruction = tokenizer(f"<|im_start|>system\nYou are a cybersecurity expert.<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_id = ds.map(process_func, remove_columns=ds.column_names)


model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda",torch_dtype=torch.bfloat16)
model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法


config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
config

model = get_peft_model(model, config)
model.print_trainable_parameters()

args = TrainingArguments(
    # Yes_or_No_10_epochs用的数据不再平衡，no多一点
    output_dir="/home/keytoliu/test_3/SCIERC/fine-tuning/checkpoint/"+data_name+"_Step5_Sentence_Have_Type_yes_or_no_15_epochs",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=5,
    logging_dir="/home/keytoliu/test_3/SCIERC/fine-tuning/checkpoint/"+data_name+"_Step5_log",
    num_train_epochs=15,
    #save_steps=100,
    bf16=True,
    save_on_each_node=True,
    gradient_checkpointing=True,
    save_total_limit=1,  # 只保留最后一个检查点
    save_strategy="epoch"  # 每个epoch保存一次
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()




