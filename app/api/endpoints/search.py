from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict
from pydantic import BaseModel
import asyncio
from app.services.search_service import perform_semantic_search
from app.services.conversation_guide_service import ConversationGuideService

router = APIRouter(
    prefix="/search",
    tags=["search"]
)

# 全局conversation guide service实例，用于保持会话状态
global_guide_service = ConversationGuideService()

class SearchQuery(BaseModel):
    """
    Model for search query input
    """
    text: str
    execute_search: bool = False  # 新增：是否执行搜索，默认为False

class SearchResultItem(BaseModel):
    """
    Single search result model
    """
    id: Optional[str]
    text: Optional[str]
    emotion_label: Optional[str]  # 处理为可选字符串，在转换时处理list类型
    score: Optional[float]

class EmotionAnalysis(BaseModel):
    """
    Model for emotion analysis results
    """
    has_emotion_content: bool
    emotion_intensity: float
    confidence: float
    needs_more_detail: bool

class EmotionStat(BaseModel):
    label: str
    count: int
    percentage: float

class EnrichedEmotionStat(BaseModel):
    """
    Single enriched emotion statistic
    """
    label: str
    count: int
    percentage: float
    definition: str
    quote: str
    analysis: str

class RAGAnalysis(BaseModel):
    """
    The main model for the entire RAG analysis package.
    """
    enriched_emotion_stats: List[EnrichedEmotionStat]
    summary_report: str

class SearchResponse(BaseModel):
    """
    Model for search response
    """
    results: List[SearchResultItem]
    message: Optional[str]
    emotion_analysis: Optional[EmotionAnalysis]
    guidance_response: Optional[str]
    rag_analysis: Optional[RAGAnalysis]
    accumulated_text: Optional[str] = None  # 返回累积文本
    input_round: Optional[int] = None  # 返回输入轮次
    ready_for_search: Optional[bool] = None  # 新增：是否准备好执行搜索

@router.post("/", response_model=SearchResponse)
async def search_emotions(query: SearchQuery):
    """
    Search for similar texts in the database and get AI analysis.
    Two-step process: 
    1. First call (execute_search=False): Quality check and guidance
    2. Second call (execute_search=True): Execute search and RAG when ready
    """
    if not query.text.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    try:
        # 使用全局service实例来保持会话状态
        guide_service = global_guide_service
        
        # 如果是执行搜索的请求，直接进行搜索（跳过重复的质量检测）
        if query.execute_search:
            print(f"Executing search for accumulated text")
            
            # 获取当前累积的文本
            accumulated_text = guide_service.get_accumulated_text()
            if not accumulated_text:
                raise HTTPException(status_code=400, detail="No accumulated text found. Please provide input first.")
            
            print(f"Searching with accumulated text: '{accumulated_text}'")
            
            # 执行搜索和RAG分析
            search_result = await asyncio.to_thread(perform_semantic_search, accumulated_text, 30)
            search_results_raw = search_result["results"]
            rag_analysis_data = search_result.get("rag_analysis")
            
            # 将原始结果转换为Pydantic模型，处理emotion_label的类型转换
            pydantic_results = []
            for doc in search_results_raw:
                emotion_label = doc.get("emotion_label")
                if isinstance(emotion_label, list):
                    emotion_label_str = str(emotion_label)
                else:
                    emotion_label_str = str(emotion_label) if emotion_label else ""
                
                pydantic_results.append(SearchResultItem(
                    id=doc.get("_id"), 
                    text=doc.get("text"), 
                    emotion_label=emotion_label_str,
                    score=doc.get("score")
                ))
            
            # 搜索完成后清空累积输入
            guide_service.clear_accumulated_input()
            
            message = "No matching documents found." if not search_results_raw else None
            return SearchResponse(
                results=pydantic_results,
                message=message,
                emotion_analysis=None,  # 搜索时不需要返回情感分析
                guidance_response="Search completed successfully. Here are the results based on your emotional experience.",
                rag_analysis=RAGAnalysis(**rag_analysis_data) if rag_analysis_data else None,
                accumulated_text=accumulated_text,
                input_round=0,  # 搜索完成后重置
                ready_for_search=False
            )
        
        # 常规质量检测流程
        print(f"Processing conversation guide analysis for: '{query.text}'")
        guide_result = await guide_service.process_user_input(query.text)
        
        # 创建情感分析响应
        emotion_analysis = None
        if guide_result and "analysis" in guide_result:
            analysis = guide_result["analysis"]
            emotion_analysis = EmotionAnalysis(
                has_emotion_content=analysis["emotion_analysis"]["has_emotion_content"],
                emotion_intensity=analysis["emotion_analysis"]["emotion_intensity"],
                confidence=analysis["emotion_analysis"]["confidence"],
                needs_more_detail=analysis.get("needs_more_detail", False),
            )
        
        # 获取状态信息
        needs_more_detail = guide_result.get("needs_more_input", True) if guide_result else True
        accumulated_text = guide_result.get("accumulated_text", "") if guide_result else ""
        input_round = guide_result.get("input_round", 0) if guide_result else 0
        ready_for_search = guide_result.get("ready_for_search", False) if guide_result else False
        
        # 调试质量检查逻辑
        print(f"Quality check details:")
        print(f"  - needs_more_detail: {needs_more_detail}")
        print(f"  - ready_for_search: {ready_for_search}")
        print(f"  - accumulated_text: '{accumulated_text}'")
        print(f"  - input_round: {input_round}")
        if guide_result and "analysis" in guide_result:
            print(f"  - sentence_count: {guide_result['analysis']['sentence_count']}")
            print(f"  - emotion_intensity: {guide_result['analysis']['emotion_analysis']['emotion_intensity']}")
        
        # 返回质量检测结果，不执行搜索
        if ready_for_search:
            message = "Your input quality is sufficient. Click the search button to find similar emotional experiences and get detailed analysis."
        else:
            message = "Please provide more detailed information about your emotional experience."
            
        return SearchResponse(
            results=[],  # 质量检测阶段不返回搜索结果
            message=message,
            emotion_analysis=emotion_analysis,
            guidance_response=guide_result.get("guidance_response") if guide_result else None,
            rag_analysis=None,  # 质量检测阶段不进行RAG分析
            accumulated_text=accumulated_text,
            input_round=input_round,
            ready_for_search=ready_for_search
        )
        
    except Exception as e:
        print(f"Unexpected error in /search endpoint: {str(e)} - Query: {query.text}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during the search."
        )

@router.post("/clear-session", response_model=Dict[str, str])
async def clear_session():
    """
    清空会话的累积输入
    """
    try:
        global_guide_service.clear_accumulated_input()
        return {"message": "Session cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")

@router.get("/session-status", response_model=Dict)
async def get_session_status():
    """
    获取当前会话状态
    """
    try:
        return {
            "accumulated_inputs": global_guide_service.accumulated_input,
            "accumulated_text": global_guide_service.get_accumulated_text(),
            "session_start_time": global_guide_service.session_start_time.isoformat(),
            "input_count": len(global_guide_service.accumulated_input)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting session status: {str(e)}")

@router.post("/execute", response_model=SearchResponse)
async def execute_search():
    """
    Execute search and RAG analysis for the accumulated text.
    This endpoint is called when user clicks the search button.
    """
    try:
        guide_service = global_guide_service
        
        # 获取当前累积的文本
        accumulated_text = guide_service.get_accumulated_text()
        if not accumulated_text:
            raise HTTPException(status_code=400, detail="No accumulated text found. Please provide input first.")
        
        print(f"Executing search for accumulated text: '{accumulated_text}'")
        
        # 执行搜索和RAG分析
        search_result = await asyncio.to_thread(perform_semantic_search, accumulated_text, 30)
        search_results_raw = search_result["results"]
        rag_analysis_data = search_result.get("rag_analysis")
        
        # 将原始结果转换为Pydantic模型，处理emotion_label的类型转换
        pydantic_results = []
        for doc in search_results_raw:
            emotion_label = doc.get("emotion_label")
            if isinstance(emotion_label, list):
                emotion_label_str = str(emotion_label)
            else:
                emotion_label_str = str(emotion_label) if emotion_label else ""
            
            pydantic_results.append(SearchResultItem(
                id=doc.get("_id"), 
                text=doc.get("text"), 
                emotion_label=emotion_label_str,
                score=doc.get("score")
            ))
        
        # 搜索完成后清空累积输入
        guide_service.clear_accumulated_input()
        
        message = "No matching documents found." if not search_results_raw else None
        return SearchResponse(
            results=pydantic_results,
            message=message,
            emotion_analysis=None,  # 搜索时不需要返回情感分析
            guidance_response="Search completed successfully. Here are the results based on your emotional experience.",
            rag_analysis=RAGAnalysis(**rag_analysis_data) if rag_analysis_data else None,
            accumulated_text=accumulated_text,
            input_round=0,  # 搜索完成后重置
            ready_for_search=False
        )
        
    except Exception as e:
        print(f"Unexpected error in /search/execute endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during search execution."
        ) 