# 用微调后的模型
import ast
import json

import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel, LoraConfig, TaskType
from datasets import Dataset

for name in ['SCIERC']:

    model_path = "/home/keytoliu/L/Qwen2.5-3B-Instruct"
    if name == 'CDTier':
        lora_path = "/home/keytoliu/test_3/fine-tuning/checkpoint/CDTier80_checkpoint_cloze/checkpoint-450"
    if name == 'SCIERC':
        lora_path = "/home/keytoliu/test_3/SCIERC/fine-tuning/checkpoint/SCIERC_Step5_Sentence_Have_Type_yes_or_no_15_epochs/checkpoint-12210"
        #lora_path = '/home/keytoliu/test_3/fine-tuning/checkpoint/v3/SCIERC_Step5_Sentence_Have_Type_yes_or_no_10_epochs/checkpoint-3700' #效果不好
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False, # 训练模式
        r=8, # Lora 秩
        lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1# Dropout 比例
    )

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16)

    # 加载lora权重
    model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)


    # 读取输入数据
    with open('/home/keytoliu/test_3/result/SCIERC/SCIERC_Judgment_Step4.json', 'r') as file:
        data_list = json.load(file)

    # 用于存储每个data对应的response和re
    results = []

    prompt = '''Is there a \"[Type]\" relationship between entities in the sentence \"[Sentence]\"?'''
    # 处理每个输入项
    for item in tqdm(data_list):

        Types = []
        result = {}

        err = eval(item['Judgment_Step4'])['err']
        maybe = []
        corr = eval(item['Judgment_Step4'])['corr']
        for type in eval(item['Judgment_Step4'])['maybe']:

            Prompt = prompt
            Prompt = Prompt.replace("[Sentence]", item['原文']).replace('[Type]', type[1])
            messages = [
                {"role": "system",
                 "content": 'You are a cybersecurity expert.'},
                {"role": "user", "content": Prompt}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=512 #100
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            if 'no' not in response.lower():
                maybe.append(type)
            else:
                err.append(type)

        result['maybe'] = maybe
        result['err'] = err
        result['corr'] = corr
        item['Judgment_Step5'] = str(result)

        # 将结果保存为JSON文件
        output_path = "/home/keytoliu/test_3/result/SCIERC/SCIERC_Judgment_Step5.json"
        with open(output_path, 'w', encoding='utf-8') as output_file:
            json.dump(data_list, output_file, ensure_ascii=False, indent=4)









