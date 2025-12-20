import random
import json
from tqdm import tqdm


for data_name in ['SCIERC']:
    data_path = 'train_triples_with_NER.json'

    prompt = '''Is there a "[Type]" relationship between entities in the sentence "[Sentence]"?'''

    relation_types = ['Used-for', 'Feature-of', 'Evaluate-for', 'Conjunction', 'Part-of', 'Hyponym-of', 'Compare']

    with open(data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)


    # 目标实体
    RE = []
    positive_samples = []
    negative_samples = []

    for step4_data in tqdm(train_data):
        # 示例文本
        text = step4_data['原文']
        types = []
        for i in step4_data['三元组']:
            if i['relation'] not in types:
                types.append(i['relation'])

        for type in types:
            Prompt = prompt
            Prompt = Prompt.replace('[Sentence]', text).replace('[Type]', type)
            positive_samples.append({'input': Prompt, 'output': 'Yes'})
        for type in relation_types:
            if type not in types:
                Prompt = prompt
                Prompt = Prompt.replace('[Sentence]', text).replace('[Type]', type)
                negative_samples.append({'input': Prompt, 'output': 'No'})

    # 合并正负样本
    RE.extend(positive_samples)
    RE.extend(negative_samples)

    # 打乱数据
    random.shuffle(RE)

    # 将结果保存为JSON文件
    output_path = "lora_data_Maturation-2.json"
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(RE, output_file, ensure_ascii=False, indent=4)