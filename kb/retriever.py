import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# 全局模型与索引缓存（提高速度）
_cached_model = None
_cached_index = None
_cached_meta = None


def load_index(index_path='kb/index/faiss.index', meta_path='kb/index/docs.pkl', model_name='all-MiniLM-L6-v2'):
    global _cached_model, _cached_index, _cached_meta

    if _cached_index is not None and _cached_meta is not None and _cached_model is not None:
        return _cached_index, _cached_meta, _cached_model

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError('Index or metadata not found. Run indexing first.')

    # 加载索引
    _cached_index = faiss.read_index(index_path)
    print("【DEBUG】加载 FAISS 索引：", index_path)

    # 加载 metadata
    with open(meta_path, 'rb') as f:
        _cached_meta = pickle.load(f)
    print("【DEBUG】加载 文档元数据：", meta_path)
    # 加载模型
    _cached_model = SentenceTransformer(model_name)

    return _cached_index, _cached_meta, _cached_model


def query(query: str, top_k=5, index_path='kb/index/faiss.index', meta_path='kb/index/docs.pkl'):
    """
    query: 查询字符串（FastAPI 中为字段名 query）
    top_k: 返回 top_k 相似文本
    """
    index, meta, model = load_index(index_path, meta_path)

    # 计算 query embedding
    q_emb = model.encode([query], convert_to_numpy=True).astype('float32')

    # 检索
    D, I = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(meta):
            continue
        results.append({
            'score': float(score),
            'source': meta[idx]['source'],
            'text': meta[idx]['text'].strip()
        })

    return results
