import os
import glob
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm


def chunk_text(text, chunk_size=500, overlap=50):
    L = len(text)
    if L == 0:
        return []

    chunks = []
    start = 0

    while start < L:
        end = min(start + chunk_size, L)
        chunks.append({'text': text[start:end], 'start': start, 'end': end})

        if end == L:
            break

        # 关键：保证往前推进，不能退回，否则死循环
        start = start + chunk_size - overlap
        if start <= 0:
            start = end

    return chunks


def index_documents(data_dir='kb/source', index_path='kb/index/faiss.index', meta_path='kb/index/docs.pkl',
                    model_name='all-MiniLM-L6-v2', chunk_size=500, overlap=50):

    os.makedirs(os.path.dirname(index_path) or '.', exist_ok=True)
    model = SentenceTransformer(model_name)
    docs_meta = []

    # 查找文件
    patterns = ['**/*.txt', '**/*.md']
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(data_dir, p), recursive=True))

    # 分块
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
                text = fh.read()
        except Exception:
            continue

        chunks = chunk_text(text, chunk_size, overlap)
        for c in chunks:
            docs_meta.append({
                'source': os.path.relpath(f),
                'text': c['text'],
                'start': c['start'],
                'end': c['end']
            })

    if not docs_meta:
        print('没有发现可索引的文档。')
        return {'status': 'no_docs'}

    texts = [d['text'] for d in docs_meta]
    batch_size = 64

    # 流式构建 index
    index = None

    for i in tqdm(range(0, len(texts), batch_size), desc='Embedding batches'):
        batch_texts = texts[i:i+batch_size]
        embs = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False).astype('float32')

        if index is None:
            dim = embs.shape[1]
            index = faiss.IndexFlatL2(dim)

        index.add(embs)

    faiss.write_index(index, index_path)

    with open(meta_path, 'wb') as f:
        pickle.dump(docs_meta, f)

    return {'status': 'ok', 'n_chunks': len(docs_meta)}
