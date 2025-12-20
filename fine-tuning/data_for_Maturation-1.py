import json
import random
from collections import defaultdict

def convert_format(input_data):
    result = []

    for item in input_data:
        # 构建原文
        text = " ".join(item["tokens"])

        # 构建NER列表
        ner_list = []
        for entity in item["entities"]:
            start = entity["start"]
            end = entity["end"]
            # 提取实体文本
            entity_text = " ".join(item["tokens"][start:end])
            ner_list.append({
                "完整名称": entity_text,
                "标签": entity["type"]
            })

        # 构建三元组列表
        triple_list = []
        for relation in item["relations"]:
            # 获取subject和object在entities中的索引
            subject_idx = relation["head"]
            object_idx = relation["tail"]

            # 获取对应的实体信息
            subject_entity = item["entities"][subject_idx]
            object_entity = item["entities"][object_idx]

            # 提取实体文本
            subject_text = " ".join(item["tokens"][subject_entity["start"]:subject_entity["end"]])
            object_text = " ".join(item["tokens"][object_entity["start"]:object_entity["end"]])

            triple_list.append({
                "subject": {"text": subject_text},
                "relation": relation["type"],
                "object": {"text": object_text}
            })

        result.append({
            "原文": text,
            "三元组": triple_list,
            "NER": ner_list
        })

    return result

data_path = 'train_triples.json'
with open(data_path, 'r', encoding='utf-8') as f:
    train_data = json.load(f)

data_path = 'SCIERC_relation_head_tail.json'
with open(data_path, 'r', encoding='utf-8') as f:
    relation_constraints = json.load(f)
relation_constraints = {
    "Part-of": [tuple(item) for item in relation_constraints["Part-of"]]
}

train_data = convert_format(train_data)


prompt = '''Please determine which of the following relationships between Entity 1 and Entity 2 in the Sentence below belongs to: ["Conjunction", "Feature-of", "Hyponym-of", "Used-for", "Part-of", "Compare", "Evaluate-for", "None"].
If there is no relationship between two entities, output None.
Sentence: [Sentence]
Entity 1: [Entity 1]
Entity 2: [Entity 2]'''

RE = []

# Step 1: 构建全局实体-类型映射
global_entity_type_map = defaultdict(set)

# 统计训练集中所有的实体及其类型
for data_item in train_data:
    for ner_item in data_item['NER']:
        global_entity_type_map[ner_item['完整名称']].add(ner_item['标签'])


# Step 2: 从原文中匹配实体
def match_entities_from_text(text, entity_type_map):
    """
    从原文中匹配实体，并根据全局实体-类型映射补充类型信息。
    优先匹配更长的实体，避免重复匹配。

    Args:
        text (str): 原文
        entity_type_map (dict): 全局实体-类型映射表
    Returns:
        dict: 匹配到的实体及其类型
    """
    matched_entities = {}
    matched_indices = set()  # 用于记录已匹配的字符索引范围

    # 按实体长度从长到短排序
    sorted_entities = sorted(entity_type_map.keys(), key=len, reverse=True)

    for entity in sorted_entities:
        start_idx = text.find(entity)

        # 如果找到了实体，并且其范围未被占用，进行匹配
        while start_idx != -1:
            end_idx = start_idx + len(entity)
            # 检查该实体范围是否已经被占用
            if not any(idx in matched_indices for idx in range(start_idx, end_idx)):
                matched_entities[entity] = list(entity_type_map[entity])
                # 标记该实体的索引范围为已占用
                matched_indices.update(range(start_idx, end_idx))
            # 查找下一个匹配的实体
            start_idx = text.find(entity, start_idx + 1)

    return matched_entities

'''
# Step 3: 生成正样本和负样本
for data_item in train_data:
    text = data_item['原文']
    # 从NER字段中获取当前句子的实体
    entities = {ner['完整名称']: ner['标签'] for ner in data_item['NER']}

    # 使用全局实体映射表对原文进行匹配
    additional_entities = match_entities_from_text(data_item['原文'], global_entity_type_map)

    # 合并实体（以补充的实体为主）
    for entity, entity_types in additional_entities.items():
        if entity not in str(entities):
            # 如果原NER中没有此实体，则补充
            entities[entity] = entity_types[0]  # 默认选择第一个类型

    # 转换为列表形式
    entity_list = list(entities.items())

    # 生成正样本数据
    positive_pairs = set()
    for item in data_item['三元组']:
        Prompt = prompt
        head = item['subject']['text']
        relation = item['relation']
        tail = item['object']['text']
        Prompt = Prompt.replace('[Sentence]', text).replace('[Entity 1]', head).replace('[Entity 2]', tail)
        RE.append({'input': Prompt, 'output': relation})
        positive_pairs.add((head, tail))

        # 生成负样本数据
        for i, (head, head_type) in enumerate(entity_list):
            for j, (tail, tail_type) in enumerate(entity_list):
                if i != j and (head, tail) not in positive_pairs:
                    # 检查头尾实体类型是否符合关系类型限制
                    valid_type_combination = any(
                        (head_type, tail_type) in constraints
                        for constraints in relation_constraints.values()
                    )
                    # 如果符合类型限制但不在正样本中，则生成负样本
                    if valid_type_combination:
                        Prompt = prompt.replace('[Sentence]', text).replace('[Entity 1]', head).replace('[Entity 2]',
                                                                                                        tail)
                        RE.append({'input': Prompt, 'output': 'None'})

# 打乱数据
random.shuffle(RE)

output_path = "/home/keytoliu/test_3/fine-tuning/data/v3/data_for_Step4_Which_Type_or_None.json"
with open(output_path, 'w', encoding='utf-8') as output_file:
    json.dump(RE, output_file, ensure_ascii=False, indent=4)
'''
# Step 3: 生成正样本和负样本
positive_samples = []
negative_samples = []

for data_item in train_data:
    text = data_item['原文']
    # 从NER字段中获取当前句子的实体
    entities = {ner['完整名称']: ner['标签'] for ner in data_item['NER']}

    # 使用全局实体映射表对原文进行匹配
    additional_entities = match_entities_from_text(data_item['原文'], global_entity_type_map)

    # 合并实体（以补充的实体为主）
    for entity, entity_types in additional_entities.items():
        if entity not in str(entities):
            # 如果原NER中没有此实体，则补充
            entities[entity] = entity_types[0]  # 默认选择第一个类型

    # 转换为列表形式
    entity_list = list(entities.items())

    # 生成正样本数据
    positive_pairs = set()
    for item in data_item['三元组']:
        Prompt = prompt
        head = item['subject']['text']
        relation = item['relation']
        tail = item['object']['text']
        Prompt = Prompt.replace('[Sentence]', text).replace('[Entity 1]', head).replace('[Entity 2]', tail)
        positive_samples.append({'input': Prompt, 'output': relation})
        positive_pairs.add((head, tail))

    # 生成所有可能的负样本数据
    for i, (head, head_type) in enumerate(entity_list):
        for j, (tail, tail_type) in enumerate(entity_list):
            if i != j and (head, tail) not in positive_pairs:
                # 检查头尾实体类型是否符合关系类型限制
                valid_type_combination = any(
                    (head_type, tail_type) in constraints
                    for constraints in relation_constraints.values()
                )
                # 如果符合类型限制但不在正样本中，则生成负样本
                if valid_type_combination:
                    Prompt = prompt.replace('[Sentence]', text).replace('[Entity 1]', head).replace('[Entity 2]', tail)
                    negative_samples.append({'input': Prompt, 'output': 'None'})

# 随机选择与正样本数量相同的负样本
num_positive_samples = len(positive_samples)
random.shuffle(negative_samples)  # 打乱负样本顺序
selected_negative_samples = negative_samples[:num_positive_samples]

# 合并正样本和随机选择的负样本
RE = positive_samples + selected_negative_samples

# 打乱数据
random.shuffle(RE)

output_path = "lora_data_Maturation-1.json"
with open(output_path, 'w', encoding='utf-8') as output_file:
    json.dump(RE, output_file, ensure_ascii=False, indent=4)