# slow-kiosk-ai

Python 기반 키오스크 LLM 엔진 템플릿입니다.

역할:
- React/브라우저에서 인식한 텍스트(브라우저 STT)를 입력으로 받고
- OpenAI LLM을 이용해 의도를 분석하여
  - assistant_text (사용자에게 보여줄/읽어줄 문장)
  - actions (장바구니 조작/화면 전환 등 액션 리스트)
- 를 JSON으로 반환합니다.

기본 엔드포인트:
- `POST /analyze`

요청 예시(JSON):

```json
{
  "text": "치즈버거 세트 하나 주세요",
  "scene": "BURGER_SELECT",
  "cart": {
    "items": []
  },
  "menu": [
    {
      "menuId": "CHEESEBURGER_SET",
      "name": "치즈버거 세트",
      "category": "BURGER",
      "price": 6500
    }
  ]
}
```

응답 예시(JSON):

```json
{
  "assistant_text": "네, 치즈버거 세트 하나 장바구니에 담았습니다.",
  "actions": [
    {
      "type": "ADD_ITEM",
      "menuId": "CHEESEBURGER_SET",
      "qty": 1
    }
  ],
  "should_finish": false
}
```

## 1. 설치

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. 환경 변수 설정

`.env` 파일을 만들고 다음 값을 채워 주세요.

```bash
cp .env.example .env
```

`.env` 내용:

```bash
OPENAI_API_KEY=YOUR_API_KEY_HERE
OPENAI_MODEL=gpt-4.1-mini
```

## 3. 서버 실행

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 4. 간단 테스트

```bash
curl -X POST "http://localhost:8000/analyze" \      -H "Content-Type: application/json" \      -d '{
    "text": "치즈버거 세트 하나 주세요",
    "scene": "BURGER_SELECT",
    "cart": { "items": [] },
    "menu": [
      {
        "menuId": "CHEESEBURGER_SET",
        "name": "치즈버거 세트",
        "category": "BURGER",
        "price": 6500
      }
    ]
  }'
```

이 템플릿을 Spring 백엔드에서 `http://slow-kiosk-ai:8000/analyze` 와 같이 호출해서
장바구니 업데이트 및 상태 전환에 사용할 수 있습니다.