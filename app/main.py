# app/main.py
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv

from .models import AnalyzeRequest, AnalyzeResponse
from .llm_client import call_llm

# .env 읽기 (OPENAI_API_KEY, OPENAI_MODEL 등)
load_dotenv()

app = FastAPI(
    title="Slow Kiosk AI Service",
    version="0.2.0",
    description="키오스크 주문 LLM 백엔드 (Python + FastAPI, 재료/커스터마이즈 지원)",
)

# CORS (로컬 프론트/백 테스트용, 필요에 따라 도메인 제한)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 실제 운영 시 특정 도메인만 허용하는 게 안전
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """
    React(STT 처리 완료 텍스트) -> Spring -> Python 으로 들어오는 메인 엔드포인트.
    """
    # 여기서 req.text, req.scene, req.cart, req.menu를 LLM에 넘겨 분석
    result = call_llm(req)
    return result
