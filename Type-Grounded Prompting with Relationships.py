from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids  # 使用 KMedoids
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import json
from collections import defaultdict

from FlagEmbedding import BGEM3FlagModel
import heapq
from typing import List, Dict, Any

BGEM3_path ='/home/keytoliu/L/bge-m3'
BGEM3_model = BGEM3FlagModel(BGEM3_path, use_fp16=True)

json_path = '/home/keytoliu/test_3/data/Joint/SCIERC_train_triples.json'
class TripleSentenceFinder:
    def __init__(self, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data: List[Dict[str, Any]] = json.load(f)

    def find_sentence_by_triple(self, triple_str: str) -> str:
        """
        triple_str: 形如 "头实体 关系 尾实体" 的字符串
        return: 该三元组所在句子的完整文本；找不到返回 ""
        """
        for sample in self.data:
            sentence = sample['text']
            triple_list = sample['triple_list']
            for triple in triple_list:
                if ' '.join(triple)==triple_str:
                    return sentence


def convert_to_triplet(input_str: str, relation: str) -> str:
    # 使用关系字符串分割输入字符串
    parts = input_str.split(relation)
    if len(parts) != 2:
        raise ValueError("输入字符串格式不正确，无法分割为两部分")

    # 去除首尾空格
    head = parts[0].strip()
    tail = parts[1].strip()

    # 组合成三元组格式
    triplet = f"({head}, {relation}, {tail})"
    return triplet


finder = TripleSentenceFinder(json_path)
# 1. 读取原始数据
# 读取 JSON 文件
with open('/home/keytoliu/test_3/data/Joint/SCIERC_ori/train_triples.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 用 set 去重
rel2sent = defaultdict(set)

for sample in data:
    tokens = sample['tokens']
    ents   = sample['entities']
    for rel in sample['relations']:
        rel_type = rel['type']
        # 取出头尾实体
        head_span = ents[rel['head']]
        tail_span = ents[rel['tail']]
        head_str  = ' '.join(tokens[head_span['start']:head_span['end']])
        tail_str  = ' '.join(tokens[tail_span['start']:tail_span['end']])
        # 拼接
        rel2sent[rel_type].add(f"{head_str} {rel_type} {tail_str}")

# 3. 转成 list 并排序（方便查看）
rel2sent = {k: sorted(v) for k, v in rel2sent.items()}

# 将 label_dict 转换为普通的列表，列表中每个元素是一个字典，包含标签和对应的实体名称列表
label_list = [{"标签": label, "实体": list(names)} for label, names in rel2sent.items()]

# 输出结果
for i in label_list:
    Prompt = '''Please analyze and provide a precise definition for the "[Relation]" relationship based on the following sentence and the triplet that contains this relationship.

[Example]
Consider the semantics and context to clarify the logical connection between the objects represented by this relationship type, and provide a clear and rigorous definition.'''
    Example = '''Sentence: [Sentence]
Triplet: [Triplet]'''

    prompt = Prompt
    prompt = prompt.replace('[Relation]', i['标签'])

    words = i['实体']
    word_embeddings_np = BGEM3_model.encode(words)['dense_vecs']

    # 使用 K-Means 聚类
    kmeans = KMeans(n_clusters=10, random_state=22)
    kmeans.fit(word_embeddings_np)

    # 获取每个单词的聚类标签
    kmeans_labels = kmeans.labels_

    # 获取聚类中心
    kmeans_centroids = kmeans.cluster_centers_

    # 计算每个单词到各个类别中心的距离
    kmeans_distances = cosine_distances(word_embeddings_np, kmeans_centroids)

    # 找出每个类别的代表性单词
    kmeans_representative_words = []
    for iii in range(kmeans.n_clusters):
        cluster_indices = np.where(kmeans_labels == iii)[0]
        cluster_distances = kmeans_distances[cluster_indices, iii]
        closest_index = cluster_indices[np.argmin(cluster_distances)]
        kmeans_representative_words.append(words[closest_index])

    all_ex = ""
    for ii in kmeans_representative_words:
        example = Example
        example = example.replace('[Sentence]', finder.find_sentence_by_triple(ii)).replace('[Triplet]', convert_to_triplet(ii,i['标签']))
        example = example + '\n'
        all_ex = all_ex + example
    prompt = prompt.replace('[Example]', all_ex)
    print(prompt)

