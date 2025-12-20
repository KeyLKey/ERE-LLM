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

BGEM3_path ='/home/keytoliu/L/bge-m3'
BGEM3_model = BGEM3FlagModel(BGEM3_path, use_fp16=True)



# 1. 读取原始数据
# 读取 JSON 文件
with open('/home/keytoliu/test_3/data/Joint/SCIERC_ori/train_triples.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 用 set 去重
type2names = defaultdict(set)

for sample in data:
    tokens = sample['tokens']
    for ent in sample['entities']:
        ent_type = ent['type']
        # 取出连续 token 并拼成空格分隔的字符串
        name = ' '.join(tokens[ent['start']:ent['end']])
        type2names[ent_type].add(name)

# 3. 把 set 转成 list（JSON 不直接支持 set）
type2names = {k: sorted(v) for k, v in type2names.items()}

# 将 label_dict 转换为普通的列表，列表中每个元素是一个字典，包含标签和对应的实体名称列表
label_list = [{"标签": label, "实体": list(names)} for label, names in type2names.items()]

# 输出结果
for i in label_list:
    print(i['标签'])
    words = i['实体']
    word_embeddings_np = BGEM3_model.encode(words)['dense_vecs']

    # 使用 K-Means 聚类
    kmeans = KMeans(n_clusters=5, random_state=22)
    kmeans.fit(word_embeddings_np)

    # 获取每个单词的聚类标签
    kmeans_labels = kmeans.labels_

    # 获取聚类中心
    kmeans_centroids = kmeans.cluster_centers_

    # 计算每个单词到各个类别中心的距离
    kmeans_distances = cosine_distances(word_embeddings_np, kmeans_centroids)

    # 找出每个类别的代表性单词
    kmeans_representative_words = []
    for i in range(kmeans.n_clusters):
        cluster_indices = np.where(kmeans_labels == i)[0]
        cluster_distances = kmeans_distances[cluster_indices, i]
        closest_index = cluster_indices[np.argmin(cluster_distances)]
        kmeans_representative_words.append(words[closest_index])

    print("K-Means 代表性单词：", kmeans_representative_words)

