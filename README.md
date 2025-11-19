# Slow Kiosk AI

키오스크 음성 주문용 AI 백엔드 서버입니다.  

- 입력:  
  - STT로 인식된 사용자 발화 텍스트 (`text`)
  - 현재 화면 상태 (`scene`)
  - 현재 장바구니 (`cart`)
  - 현재 화면에서 선택 가능한 메뉴 목록 (`menu`)
- 출력:  
  - 키오스크가 말할 멘트 (`assistant_text`)
  - 장바구니 변경 액션 리스트 (`actions`)
  - 주문 종료 여부 (`should_finish`)
  - 다음 화면 상태 (`next_scene`)

> 결제/주문 저장/DB 연동 등은 Spring 백엔드에서 처리 합니다.

---

## 1. 폴더 구조

```bash
slow-kiosk-ai/
├─ app/
│  ├─ __init__.py
│  ├─ main.py          # FastAPI 엔트리포인트 (/health, /analyze)
│  ├─ llm_client.py    # OpenAI LLM 호출 + 프롬프트 구성
│  ├─ models.py        # Request/Response/메뉴/액션 모델 정의
├─ burger_menu_master_with_ingredients.csv  # 예시 메뉴 DB
├─ requirements.txt
├─ .env                # OpenAI 키 등 환경변수 (gitignore 대상)
└─ README.md
2. 실행 방법
2.1. 가상환경 생성 및 패키지 설치
bash
코드 복사
python -m venv .venv
.venv\Scripts\activate      # Windows 기준

pip install -r requirements.txt
2.2. 환경변수 설정 (.env)
프로젝트 루트에 .env 파일 생성:

env
OPENAI_API_KEY=sk-...       # 본인 OpenAI API 키
OPENAI_MODEL=gpt-4.1-mini   # (선택) 기본 사용 모델


2.3. 서버 실행
bash

uvicorn app.main:app --reload
Swagger 문서: http://127.0.0.1:8000/docs

헬스체크: GET http://127.0.0.1:8000/health

3. 주요 엔드포인트
3.1. GET /health
서버 상태 확인용

예시 응답

json

{ "status": "ok" }
3.2. POST /analyze
키오스크 대화/주문 로직을 담당하는 메인 API

요청 예시

json
{
  "text": "스테디 와퍼 세트 하나에 콜라 제로로 주세요. 피클은 빼주세요.",
  "scene": "SELECT_BURGER",
  "cart": { "items": [] },
  "menu": [
    {
      "menuId": "B001",
      "name": "스테디 와퍼",
      "category": "BURGER",
      "price": 6900,
      "tags": ["대표메뉴", "와퍼", "소고기"],
      "ingredients_ko": "참깨빵, 양상추, 양파, 피클, 소고기 패티, 케첩, 마요네즈",
      "customizable_ko": "피클, 양파, 소스, 치즈, 베이컨"
    }
  ]
}
응답 예시

json
{
  "assistant_text": "스테디 와퍼 세트 1개를 담고, 피클은 빼드릴게요. 다른 메뉴도 추가하시겠어요?",
  "actions": [
    {
      "type": "ADD_ITEM",
      "menuId": "B001",
      "qty": 1,
      "customize": null
    },
    {
      "type": "CUSTOMIZE",
      "menuId": "B001",
      "qty": 1,
      "customize": {
        "add": [],
        "remove": ["피클"]
      }
    }
  ],
  "should_finish": false,
  "next_scene": "SELECT_SIDE"
}
