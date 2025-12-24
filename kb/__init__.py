import os
import json
from typing import List, Dict, Any
from datetime import datetime
from .auto_writer import write_alarm_case_to_kb
from .retriever import query as kb_query
from .indexing import index_documents

class KnowledgeBase:
    """知识库管理类"""
    def __init__(self, base_dir="kb"):
        self.base_dir = base_dir
        self.source_dir = os.path.join(base_dir, "source")
        self.index_dir = os.path.join(base_dir, "index")
        self.cases_dir = os.path.join(base_dir, "cases")
        
        # 创建目录
        for dir_path in [self.source_dir, self.index_dir, self.cases_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def add_case(self, case_data: Dict[str, Any]) -> str:
        """添加报警案例到知识库"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        case_id = f"case_{timestamp}"
        
        case_file = os.path.join(self.cases_dir, f"{case_id}.json")
        
        # 添加元数据
        case_data.update({
            "case_id": case_id,
            "created_at": datetime.now().isoformat(),
            "reviewed": False,  # 标记是否需要人工复核
            "review_result": None
        })
        
        with open(case_file, 'w', encoding='utf-8') as f:
            json.dump(case_data, f, ensure_ascii=False, indent=2)
        
        # 同时写入Markdown格式（供索引）
        write_alarm_case_to_kb(case_data)
        
        return case_id
    
    def get_similar_cases(self, query: str, top_k: int = 5, 
                         similarity_threshold: float = 0.7) -> List[Dict]:
        """获取相似案例"""
        try:
            # 使用导入的 kb_query 函数
            results = kb_query(query, top_k=top_k)
            
            # 过滤相似度
            filtered = [r for r in results if r['score'] < similarity_threshold]
            return filtered
        except Exception as e:
            print(f"【ERROR】查询知识库失败: {e}")
            return []
    
    def update_index(self):
        """更新知识库索引"""
        try:
            result = index_documents(data_dir=self.source_dir)
            return result
        except Exception as e:
            print(f"【ERROR】更新索引失败: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_statistics(self) -> Dict:
        """获取知识库统计信息"""
        stats = {
            "total_cases": len([f for f in os.listdir(self.cases_dir) if f.endswith('.json')]) if os.path.exists(self.cases_dir) else 0,
            "total_documents": len([f for f in os.listdir(self.source_dir) if f.endswith(('.md', '.txt'))]) if os.path.exists(self.source_dir) else 0,
            "index_exists": os.path.exists(os.path.join(self.index_dir, "faiss.index")) if os.path.exists(self.index_dir) else False,
            "last_update": None
        }
        
        # 获取最后更新时间
        index_file = os.path.join(self.index_dir, "faiss.index")
        if os.path.exists(index_file):
            stats["last_update"] = datetime.fromtimestamp(
                os.path.getmtime(index_file)
            ).strftime("%Y-%m-%d %H:%M:%S")
        
        return stats

# 创建全局知识库实例
kb_instance = KnowledgeBase()

# 导出常用别名
KnowledgeBase = KnowledgeBase
kb = kb_instance