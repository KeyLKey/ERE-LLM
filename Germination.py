import heapq
import re

from tqdm import tqdm
from prompts import ModelPrompts
import json
from openai import OpenAI
from FlagEmbedding import BGEM3FlagModel
BGEM3_path ='/home/keytoliu/L/bge-m3'
BGEM3_model = BGEM3FlagModel(BGEM3_path, use_fp16=True)

def top_three_with_indices(lst,n):
    # 使用 heapq.nlargest 获取最大n个数字及其位置
    top_three = heapq.nlargest(n, enumerate(lst), key=lambda x: x[1])
    # top_three 是一个包含 (index, value) 的元组列表
    indices = [index for index, value in top_three]
    values = [value for index, value in top_three]
    return indices, values

model_prompts = ModelPrompts()
client = OpenAI(api_key="sk-ebd8fd268b7e47a8b39bd5d1486550b2", base_url="https://api.deepseek.com")

for data_name in ['SCIERC']:
    data_path = "/home/keytoliu/test_3/data/Joint/SCIERC_test_triples_without_null.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    data_path = "/home/keytoliu/test_3/data/Joint/SCIERC_train_triples_without_null.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
        
    Types = []
    if 'SCIERC' in data_path:
        Types = ['Used-for', 'Feature-of', 'Evaluate-for', 'Conjunction', 'Part-of', 'Hyponym-of', 'Compare']
    if 'STIXnet' in data_path:
        Types = ['used-by', 'uses', 'targets', 'originates-from', 'deliver', 'targeted-by', 'located-at', 'attributed-to']
    if 'LADDER' in data_path:
        Types = ['isA', 'targets', 'uses', 'hasAuthor', 'has', 'variantOf', 'hasAlias', 'indicates', 'discoveredIn', 'exploits']
    if 'CDTier' in data_path:
        Types = ['related_to', 'operating_in', 'consist_of', 'target_at', 'uses', 'develope', 'originated_from', 'alias_of', 'cooperate_with', 'kin', 'launch']



    # 提取每个条目的 '句子' 字段
    Text = [item['text'] for item in train_data]
    Triple = [item['triple_list'] for item in train_data]

    Triple1 = [] # 句子
    Triple2 = [] # 实体
    Triple3 = [] # 联合
    for item in train_data:
        Triple1.append(item['text'])
        tri0 = ''
        for triple in item["triple_list"]:
            tri0 = tri0 + ' '.join(triple) + '. '
        Triple2.append(tri0)
        tri = ''
        for triple in item["triple_list"]:
            tri = tri + ' '.join(triple) + '. '
        tri = tri + item['text']
        Triple3.append(tri)

    embeddings_sentences1 = BGEM3_model.encode(Triple1, return_dense=True, return_sparse=True,return_colbert_vecs=True)
    embeddings_sentences2 = BGEM3_model.encode(Triple2, return_dense=True, return_sparse=True, return_colbert_vecs=True)
    embeddings_sentences3 = BGEM3_model.encode(Triple3, return_dense=True, return_sparse=True, return_colbert_vecs=True)

    prompt_class = 'Step1'
    prompt = model_prompts.get_prompt(prompt_class)

    results = []
    for item in tqdm(test_data):
        text = item.get("text")  # 获取原文
        label = item.get("triple_list")
        embeddings_targe = BGEM3_model.encode(text, return_dense=True, return_sparse=True, return_colbert_vecs=True)
        Prompt = prompt
        Prompt = Prompt.replace("[Sentence]",item['text'])

        scores1 = []
        scores2 = []
        scores3 = []
        for i in range(len(Triple1)):
            score = BGEM3_model.colbert_score(embeddings_targe['colbert_vecs'],embeddings_sentences1['colbert_vecs'][i])
            scores1.append(score)
            score = BGEM3_model.colbert_score(embeddings_targe['colbert_vecs'],embeddings_sentences2['colbert_vecs'][i])
            scores2.append(score)
            score = BGEM3_model.colbert_score(embeddings_targe['colbert_vecs'],embeddings_sentences3['colbert_vecs'][i])
            scores3.append(score)
        indices1, _ = top_three_with_indices(scores1, 1)  # 这里只取最高分的一个索引
        indices2, _ = top_three_with_indices(scores2, 1)
        indices3, _ = top_three_with_indices(scores3, 1)
        index1, index2, index3 = indices1[0], indices2[0], indices3[0]

        Example1_Step1 = set()
        for i in Triple[index1]:
            Example1_Step1.add(i[0])
            Example1_Step1.add(i[2])
        Example1_Step1 = list(Example1_Step1)
        Example1_Step1 = json.dumps(Example1_Step1)
        Prompt = Prompt.replace("[Example Sentence1]", Text[index1])
        Prompt = Prompt.replace("[Example1 Step1]", Example1_Step1)
        Prompt = Prompt.replace("[Example1 Step2]", str(Triple[index1]))

        if index1==index2:
            Prompt = re.sub(r'Example2:.*?Output:\s*\[Example2 Step2\]\s*', '', Prompt, flags=re.DOTALL)
        else:
            Example2_Step1 = set()
            for i in Triple[index2]:
                Example2_Step1.add(i[0])
                Example2_Step1.add(i[2])
            Example2_Step1 = list(Example2_Step1)
            Example2_Step1 = json.dumps(Example2_Step1)
            Prompt = Prompt.replace("[Example Sentence2]", Text[index2])
            Prompt = Prompt.replace("[Example2 Step1]", Example2_Step1)
            Prompt = Prompt.replace("[Example2 Step2]", str(Triple[index2]))

        if index2==index3:
            Prompt = re.sub(r'Example3:.*?Output: \[Example3 Step2\]', '', Prompt, flags=re.DOTALL)
        elif index3==index1:
            Prompt = re.sub(r'Example3:.*?Output: \[Example3 Step2\]', '', Prompt, flags=re.DOTALL)
        else:
            Example3_Step1 = set()
            for i in Triple[index3]:
                Example3_Step1.add(i[0])
                Example3_Step1.add(i[2])
            Example3_Step1 = list(Example3_Step1)
            Example3_Step1 = json.dumps(Example3_Step1)
            Prompt = Prompt.replace("[Example Sentence3]", Text[index3])
            Prompt = Prompt.replace("[Example3 Step1]", Example3_Step1)
            Prompt = Prompt.replace("[Example3 Step2]", str(Triple[index3]))

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": Prompt},
            ],
            stream=False
        )

        text0 = response.choices[0].message.content

        # 1. 字符串整体反转
        try:
            rev = text0[::-1]
            # 2. 在反转后的字符串里找第一个 "]]...[["
            m = re.search(r'\]\](.*?)\[\[', rev)
            if m:
                # 3. 把匹配到的内容再反转回原顺序，并补上双方括号
                result = '[[' + m.group(1)[::-1] + ']]'
            else:
                if '[]' in text0:
                    result = "[]"
                else:
                    result = text0
        except Exception:
            result = text0

        results.append({
            '原文': text,
            "label": str(label),
            "step1": result,
        })

        # 将结果保存为JSON文件
        output_path = "/home/keytoliu/test_3/result/SCIERC/" + data_name+"_"+prompt_class+".json"
        with open(output_path, 'w', encoding='utf-8') as output_file:
            json.dump(results, output_file, ensure_ascii=False, indent=4)




