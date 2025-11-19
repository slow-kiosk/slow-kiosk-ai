from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .models import AnalyzeRequest, AnalyzeResponse
from .llm_client import analyze_with_llm


app = FastAPI(
    title="Slow Kiosk AI Service",
    description="키오스크 주문 LLM 백엔드 템플릿 (Python + FastAPI)",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    """간단한 헬스 체크 엔드포인트."""
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """사용자 발화 + 현재 상태를 받아 LLM으로 의도 분석을 수행합니다."""
    result = analyze_with_llm(req)
    # FastAPI가 Pydantic 모델을 알아서 JSON으로 변환해 주지만,
    # 명시적으로 JSONResponse를 사용해도 됩니다.
    return JSONResponse(content=result.model_dump())