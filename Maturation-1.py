# 用微调后的模型
import heapq
import json

import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel, LoraConfig, TaskType
from datasets import Dataset

def top_three_with_indices(lst,n):
    # 使用 heapq.nlargest 获取最大n个数字及其位置
    top_three = heapq.nlargest(n, enumerate(lst), key=lambda x: x[1])
    # top_three 是一个包含 (index, value) 的元组列表
    indices = [index for index, value in top_three]
    values = [value for index, value in top_three]
    return indices, values
from FlagEmbedding import BGEM3FlagModel
BGEM3_path ='/home/keytoliu/L/bge-m3'
BGEM3_model = BGEM3FlagModel(BGEM3_path, use_fp16=True)
model_name = "/home/keytoliu/L/Qwen2.5-3B-Instruct"
data_name = 'SCIERC'
two_entity_data = []


for name in ['SCIERC']:
    model_path = "/home/keytoliu/L/Qwen2.5-3B-Instruct"
    if name == 'SCIERC':
        #lora_path = '/home/keytoliu/test_3/SCIERC/fine-tuning/checkpoint/SCIERC_Step4_Which_Type_or_None_15_epochs/checkpoint-6030'
        lora_path = '/home/keytoliu/test_3/SCIERC/fine-tuning/checkpoint/New_SCIERC_Step4_Which_Type_or_None_15_epochs/checkpoint-6045'
        #lora_path = '/home/keytoliu/test_3/SCIERC/fine-tuning/checkpoint/SCIERC_Step4_Which_Type_or_None_5_epochs/checkpoint-2015'
        relation_types = ["Conjunction", "Feature-of", "Hyponym-of", "Used-for", "Part-of", "Compare", "Evaluate-for"]
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
    with open('/home/keytoliu/test_3/result/SCIERC/SCIERC_New_Find.json', 'r', encoding='utf-8') as f:
        data_list = json.load(f)  # 「实体-类型」映射
    # 用于存储每个data对应的response和re
    results = []
    Prompt = '''Please determine which of the following relationships between Entity 1 and Entity 2 in the Sentence below belongs to: ["Conjunction", "Feature-of", "Hyponym-of", "Used-for", "Part-of", "Compare", "Evaluate-for", "None"].
If there is no relationship between two entities, output None.
Sentence: [Sentence]
Entity 1: [Entity 1]
Entity 2: [Entity 2]'''
    # 处理每个输入项
    for item in tqdm(data_list):
        #data = item.get("instruction")
        real = item.get("label")
        result = {}
        err = []
        maybe = []
        corr = []
        # 处理可能值
        prediction = []
        for i in eval(item['step1']):
            prediction.append(i)
        for i in eval(item['New_Find']):
            prediction.append(i)
        for i in prediction:
            head_entity = i[0]
            tail_entity = i[2]
            two_scores_head_entity = []
            prompt = Prompt
            prompt = prompt.replace('[Sentence]', item['原文']).replace('[Entity 1]', i[0]).replace('[Entity 2]', i[2])
            messages = [
                {"role": "system",
                 "content": 'You are a cybersecurity expert.'},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=512  # 100
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            if 'none' in response.lower():
                err.append(i)
            elif response.lower()==i[1].lower():
                corr.append(i)
            else:
                maybe.append(i)

        result['err']=err
        result['maybe']=maybe
        result['corr']=corr
        item['Judgment_Step4'] = str(result)
        output_path = "/home/keytoliu/test_3/result/SCIERC/" + data_name + "_Judgment_Step4.json"
        with open(output_path, 'w', encoding='utf-8') as output_file:
            json.dump(data_list, output_file, ensure_ascii=False, indent=4)



