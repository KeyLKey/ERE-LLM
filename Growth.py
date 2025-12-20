import ast
import re

from openai import OpenAI
import json

from tqdm import tqdm

from prompts import ModelPrompts
import heapq
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import json
import re
from collections import defaultdict
from FlagEmbedding import BGEM3FlagModel

client = OpenAI(api_key="sk-ebd8fd268b7e47a8b39bd5d1486550b2", base_url="https://api.deepseek.com")
model_prompts = ModelPrompts()
BGEM3_path ='/home/keytoliu/L/bge-m3'
BGEM3_model = BGEM3FlagModel(BGEM3_path, use_fp16=True)

def top_three_with_indices(lst,n):
    # 使用 heapq.nlargest 获取最大n个数字及其位置
    top_three = heapq.nlargest(n, enumerate(lst), key=lambda x: x[1])
    # top_three 是一个包含 (index, value) 的元组列表
    indices = [index for index, value in top_three]
    values = [value for index, value in top_three]
    return indices, values

# 加载固定名词的函数
def load_fixed_nouns(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

# 自定义词性标注函数
def custom_pos_tag(words, fixed_nouns):
    pos_tags = pos_tag(words)
    custom_tags = []
    for word, tag in pos_tags:
        if word in fixed_nouns:
            custom_tags.append((word, 'NN'))  # 强制将固定名词标注为NN
        else:
            custom_tags.append((word, tag))  # 使用默认的词性标注
    return custom_tags

from collections import Counter


def collect_frequent_entity_names(json_file_path, min_count=10):
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 统计所有实体名称的出现次数
    entity_name_counter = Counter()

    for item in data:
        tokens = item.get('tokens', [])
        for entity in item.get('entities', []):
            # 提取实体名称（根据start和end索引从tokens中获取）
            start = entity.get('start', 0)
            end = entity.get('end', 0)
            entity_name = ' '.join(tokens[start:end])

            if entity_name:
                entity_name_counter[entity_name] += 1

    # 收集出现次数大于min_count的实体名称
    frequent_entity_names = [
        entity_name for entity_name, count in entity_name_counter.items()
        if count > min_count
    ]

    return frequent_entity_names, dict(entity_name_counter)

for data_name in ['SCIERC']:
    data_path = "/home/keytoliu/test_3/data/Joint/SCIERC_train_triples_without_null.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    step2_path = '/home/keytoliu/test_3/result/SCIERC/SCIERC_Step1.json'
    with open(step2_path, 'r', encoding='utf-8') as f:
        step2_datas = json.load(f)
    with open('/home/keytoliu/test_3/data/Joint/SCIERC_entity2types.json', 'r', encoding='utf-8') as f:
        entity2types = json.load(f)  # 「实体-类型」映射
    with open('/home/keytoliu/test_3/data/Joint/SCIERC_relation_head_tail.json', 'r', encoding='utf-8') as f:
        relation_head_tail = json.load(f)  # 「实体-类型」映射
    with open('/home/keytoliu/test_3/data/Joint/SCIERC_ori/train_triples.json', 'r', encoding='utf-8') as f:
        ori_train_triples = json.load(f)
    frequent_entities, all_counts = collect_frequent_entity_names('/home/keytoliu/test_3/data/Joint/SCIERC_ori/train_triples.json', min_count=5)
    entity_data = sorted(frequent_entities, key=len, reverse=True) # 将长的放前面先判断

    prompt_class = 'Step3'
    prompt = model_prompts.get_prompt(prompt_class)
    if 'LADDER' in data_name:
        Types = ['isA', 'targets', 'uses', 'hasAuthor', 'has', 'variantOf', 'hasAlias', 'indicates', 'discoveredIn', 'exploits']
    if 'SCIERC' in data_name:
        Types = ['Used-for', 'Feature-of', 'Evaluate-for', 'Conjunction', 'Part-of', 'Hyponym-of', 'Compare']


    # 目标实体
    for step2_data in tqdm(step2_datas):
        # 示例文本
        new_find = []
        text = step2_data['原文']
        step2_result = step2_data['step1']
        step2_result = ast.literal_eval(step2_result)
        already_have = set()
        maybe_have = []
        for tri in step2_result:
            already_have.add(tri[0])
            already_have.add(tri[2])
        already_have = list(already_have)

        for entity in entity_data:
            if entity in already_have:
                continue
            if entity in text:
                maybe_have.append(entity)
                text = text.replace(entity, '')
        for target_entity in maybe_have:
            text = step2_data['原文']
            target_sentence = text
            
            prompt_1 = '''You are an expert in information extraction. Now, please determine whether the entity "[Entity]" in the sentence "[Sentence]" has any of the following relationships with other entities: ['Conjunction', 'Feature-of', 'Hyponym-of', 'Used-for', 'Part-of', 'Compare', 'Evaluate-for'].

Sentence: [Sentence]
Entity: [Entity]

If a relationship exists, output "Yes". If not, output "None". No explanation is needed.'''
            prompt_1 = prompt_1.replace('[Entity]', target_entity).replace('[Sentence]', target_sentence)
            response_1 = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt_1},
                ],
                stream=False
            )
            answer_1 = response_1.choices[0].message.content
            if 'none' in answer_1.lower():
                continue
            else:
                result0 = [] # 存储三元组中包含该实体的数据
                for item in train_data:
                    for triple in item["triple_list"]:
                        if target_entity == triple[0] or target_entity == triple[2]:
                            result0.append(item)  # 如果找到，将整个数据项添加到结果中
                            break

                result1 = [] # 存储对应的句子
                result2 = [] # 存储对应的三元组
                for item in result0:
                    triple_result=[]
                    result1.append(item['text'])
                    for triple in item["triple_list"]:
                        if triple[0] == target_entity or triple[2] == target_entity:
                            triple_result.append(triple)
                    result2.append(triple_result)
                embeddings_sentences = BGEM3_model.encode(result1, return_dense=True, return_sparse=True,return_colbert_vecs=True)  # 句子
                embeddings_targe = BGEM3_model.encode(target_sentence, return_dense=True, return_sparse=True, return_colbert_vecs=True)
                scores = []
                for i in range(len(result1)):
                    score = BGEM3_model.colbert_score(embeddings_targe['colbert_vecs'], embeddings_sentences['colbert_vecs'][i])
                    scores.append(score)
                indices, _ = top_three_with_indices(scores, 3)  # 这里只取最高分的一个索引

                prompt_2 = '''You are an expert in information extraction. Now, you will be given a sentence and an entity. Please extract the triplets containing the entity from the sentence, with the relationships in the triplets being within the scope of ["Conjunction", "Feature-of", "Hyponym-of", "Used-for", "Part-of", "Compare", "Evaluate-for"].
                
Below are the triplets corresponding to entity "[Entity]" in other sentences: 
Sentence1: [Example Sentence1]
Triplet1: [Example Output1]
Sentence2: [Example Sentence2]
Triplet2: [Example Output2]
Sentence3: [Example Sentence3]
Triplet3: [Example Output3]

Input:
Sentence: [Sentence]
Entity: [Entity]
If the entity does not have any of the above relationships in the sentence, directly output "None". Otherwise, directly output the triplets containing the entity in the format [[Head Entity 1, Relationship 1, Tail Entity 1],...]. No explanation is needed.'''
                for i in range(min(3, len(result1))):
                    prompt_2 = prompt_2.replace('[Example Sentence'+str(i+1)+']',result1[indices[i]]).replace('[Example Output'+str(i+1)+']',str(result2[indices[i]]))
                prompt_2 = prompt_2.replace('[Sentence]',step2_data['原文']).replace('[Entity]',target_entity)
                prompt_2 = prompt_2.replace('''Sentence2: [Example Sentence2]
Triplet2: [Example Output2]''','')
                prompt_2 = prompt_2.replace('''Sentence3: [Example Sentence3]
Triplet3: [Example Output3]''', '')
                response_2 = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": prompt_2},
                    ],
                    stream=False
                )

                answer_2 = response_2.choices[0].message.content
                if 'none' in answer_2.lower():
                    new_find = []
                else:
                    try:
                        for i in ast.literal_eval(answer_2):
                            new_find.append(i)
                    except:
                        new_find.append(answer_2)
        step2_data['New_Find'] = str(new_find)
        output_path = "/home/keytoliu/test_3/result/SCIERC/" + data_name + "_New_Find.json"
        with open(output_path, 'w', encoding='utf-8') as output_file:
            json.dump(step2_datas, output_file, ensure_ascii=False, indent=4)


            





