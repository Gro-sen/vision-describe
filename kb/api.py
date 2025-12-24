from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from datetime import datetime
import os
import json

router = APIRouter(prefix="/kb", tags=["knowledge-base"])
from kb import KnowledgeBase

kb = KnowledgeBase()

@router.get("/stats")
async def get_kb_stats():
    """获取知识库统计信息"""
    return kb.get_statistics()

@router.get("/search")
async def search_cases(query: str, top_k: int = 5, threshold: float = 0.7):
    """搜索相似案例"""
    results = kb.get_similar_cases(query, top_k, threshold)
    return {"query": query, "results": results, "count": len(results)}

@router.post("/add-case")
async def add_case(case_data: Dict[str, Any]):
    """添加新的报警案例"""
    required_fields = ["scene_summary", "alarm_level", "alarm_reason"]
    for field in required_fields:
        if field not in case_data:
            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
    
    case_id = kb.add_case(case_data)
    return {"status": "success", "case_id": case_id}

@router.post("/review-case")
async def review_case(case_id: str, review_result: str):
    """人工复核案例"""
    case_file = os.path.join(kb.cases_dir, f"{case_id}.json")
    
    if not os.path.exists(case_file):
        raise HTTPException(status_code=404, detail="Case not found")
    
    with open(case_file, 'r', encoding='utf-8') as f:
        case_data = json.load(f)
    
    case_data["reviewed"] = True
    case_data["review_result"] = review_result
    case_data["reviewed_at"] = datetime.now().isoformat()
    
    with open(case_file, 'w', encoding='utf-8') as f:
        json.dump(case_data, f, ensure_ascii=False, indent=2)
    
    return {"status": "success", "case_id": case_id}

@router.post("/rebuild-index")
async def rebuild_index():
    """重建知识库索引"""
    result = kb.update_index()
    return result