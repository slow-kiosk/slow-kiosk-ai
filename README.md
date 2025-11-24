# Slow Kiosk AI Service 🍔

한국형 패스트푸드 키오스크를 위한 **AI 주문 도우미 백엔드 (Python + FastAPI)** 입니다.

역할 한 줄 요약:

> React(브라우저 STT/TTS) + Spring(비즈니스 로직/주문/DB) 사이에서  
> **사용자 발화 + 현재 화면(scene) + 장바구니(cart) + 메뉴(menu)** 를 보고  
> “무슨 말을 할지, 어떤 메뉴를 담고/빼고/커스터마이즈할지, 다음 화면은 어디로 갈지”를  
> OpenAI LLM으로 판단해서 JSON으로 돌려주는 **주문 전용 AI 뇌**입니다.

---

## 1. 전체 아키텍처 개요

### 1) React (프론트)

- 브라우저 **STT**로 음성을 텍스트로 변환
- Spring으로 텍스트 전송
- Spring에서 내려준 `assistant_text`를 화면에 표시 + **브라우저 TTS**로 읽어줌

### 2) Spring (기존 백엔드)

- 화면 상태(`scene`) 관리
- 장바구니/주문(`cart`) 및 메뉴(`menu`) 관리
- `text + scene + cart + menu`를 묶어서 **Python `/analyze` 호출**
- Python이 던져준:
  - `assistant_text` → React로 전달 (TTS/UI)
  - `actions[]` → 장바구니/옵션 업데이트
  - `next_scene` → 다음 화면 전환
  - `should_finish` → 결제 단계로 넘어갈지 여부

### 3) Python (이 레포지토리)

- FastAPI + OpenAI (gpt-4.1-mini 등)
- **상태를 들고 있지 않는 stateless AI 서버**
- 입력: `AnalyzeRequest` (text, scene, cart, menu[])
- 출력: `AnalyzeResponse` (assistant_text, actions[], should_finish, next_scene)
- 메뉴 CSV/DB에서 내려온:
  - 재료(ingredients)
  - 커스터마이즈 옵션(customizable)
  - 영양 정보(kcal, 단백질/지방/당/나트륨 등)
  - 알레르기 정보(밀/대두/우유/계란/소고기/돼지고기/새우 등)
  를 참고해서,
  - “메뉴 추천”
  - “성분/영양 설명”
  - “알레르기 주의 안내”
  - “야채/소스 커스터마이즈”
  

---

## 2. 실행 방법

### 2.1. 요구 사항

- Python 3.10+
- 가상환경(venv) 권장

### 2.2. 설치

```bash
# 1) 가상환경 생성
python -m venv .venv

# 2) 활성화 (Windows PowerShell 기준)
.venv\Scripts\activate

# 3) 패키지 설치
pip install -r requirements.txt
2.3. .env 설정
프로젝트 루트에 .env 파일 생성:

env

OPENAI_API_KEY=sk-xxx_your_key_here
# 선택: 기본 모델 지정 (미지정 시 gpt-4.1-mini 사용)
OPENAI_MODEL=gpt-4.1-mini
2.4. 서버 실행


uvicorn app.main:app --reload
기본 포트: http://127.0.0.1:8000

자동 문서: http://127.0.0.1:8000/docs (Swagger)

OpenAPI JSON: http://127.0.0.1:8000/openapi.json

3. API 개요
3.1. health 체크
http

GET /health
Response: {"status": "ok"}

3.2. 주문/대화 분석
http

POST /analyze
Content-Type: application/json
Request Body: AnalyzeRequest


{
  "text": "스테디 와퍼 세트 하나에 콜라 제로로 주세요. 피클은 빼주세요.",
  "scene": "SELECT_BURGER",
  "cart": {
    "items": [
      { "menuId": "B001", "qty": 1 }
    ]
  },
  "menu": [
    {
      "menuId": "B001",
      "name": "스테디 와퍼 세트",
      "category": "SET",
      "price": 8900,
      "tags": ["대표메뉴", "와퍼", "소고기"],
      "ingredients_ko": "참깨빵, 양상추, 양파, 토마토, 피클, 소고기 패티, 케첩, 마요네즈",
      "customizable_ko": "피클, 양파, 소스, 치즈",
      "kcal": 720,
      "protein_g": 40.0,
      "fat_g": 30.0,
      "saturated_fat_g": 10.0,
      "carbs_g": 70.0,
      "sugars_g": 20.0,
      "sodium_mg": 720,
      "allergens_ko": "밀, 대두, 우유, 계란, 소고기",
      "allergens_en": "wheat, soy, milk, egg, beef",
      "allergen_wheat": true,
      "allergen_egg": true,
      "allergen_milk": true,
      "allergen_soy": true,
      "allergen_peanut": false,
      "allergen_nut": false,
      "allergen_fish": false,
      "allergen_shellfish": false,
      "allergen_pork": false,
      "allergen_beef": true,
      "allergen_shrimp": false,
      "nutrition_summary_ko": "1회 제공량 기준 약 720kcal, 단백질 40g, 지방 30g, 탄수화물 70g, 당류 20g, 나트륨 720mg 정도의 영양 정보를 가지고 있습니다.",
      "allergy_warning_ko": "이 메뉴는 밀, 대두, 우유, 계란, 소고기를(를) 함유하고 있어 알레르기가 있는 고객은 섭취에 주의가 필요합니다."
    }
  ]
}
Response Body: AnalyzeResponse


{
  "assistant_text": "스테디 와퍼 세트 1개와 콜라 제로로 담아드렸고, 와퍼에서는 피클을 빼드릴게요. 이제 사이드 메뉴도 하나 고르시겠어요?",
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
4. Request/Response 상세 스펙
4.1. AnalyzeRequest
루트 필드
Field	Type	Required	설명
text	string	Y	브라우저 STT 결과 (사용자 발화 텍스트)
scene	string	Y	현재 화면/상태 (GREETING, SELECT_BURGER, CUSTOMIZE_BURGER, SELECT_SIDE, SELECT_DRINK, CONFIRM 등)
cart	object	Y	현재 장바구니 정보
menu	array<MenuItem>	Y	현재 화면에서 선택 가능한 메뉴 리스트

cart
Field	Type	Required	설명
cart.items	array<object>	Y	장바구니에 담긴 메뉴 목록
cart.items[].menuId	string	Y	메뉴 ID (menu[].menuId와 동일)
cart.items[].qty	integer	Y	수량 (기본 1)

menu[] (MenuItem)
영양/알레르기 관련 필드는 Optional 이기 때문에,
Spring에서 준비된 것만 채워서 보내면 됩니다.

필수:

Field	Type	Required	설명
menuId	string	Y	메뉴 고유 ID
name	string	Y	메뉴 한글 이름
category	string	Y	상위 카테고리 (BURGER / SET / SIDE / DRINK / DESSERT 등)
price	integer	Y	가격(원)

선택(재료/커스터마이즈/태그):

Field	Type	Required	설명
tags	array<string>	N	추천/검색용 태그 (["대표메뉴", "와퍼", "소고기"])
ingredients_ko	string	N	재료 목록 ("참깨빵, 양상추, 양파, 피클, 소고기 패티, 케첩, 마요네즈")
customizable_ko	string	N	조절 가능한 재료/옵션 ("피클, 양파, 소스, 치즈")

선택(영양):

Field	Type	Required	설명
kcal	int	N	칼로리(kcal)
protein_g	float	N	단백질(g)
fat_g	float	N	지방(g)
saturated_fat_g	float	N	포화지방(g)
carbs_g	float	N	탄수화물(g)
sugars_g	float	N	당류(g)
sodium_mg	int	N	나트륨(mg)
nutrition_summary_ko	string	N	한글 영양 요약 문장

선택(알레르기):

Field	Type	Required	설명
allergens_ko	string	N	알레르기 유발 성분 한글 리스트 ("밀, 대두, 우유, 계란")
allergens_en	string	N	알레르기 유발 성분 영문 리스트
allergy_warning_ko	string	N	알레르기 주의 문구

플래그(있으면 LLM이 참고, 없어도 동작):

Field	Type	Required	설명
allergen_wheat	bool	N	밀
allergen_egg	bool	N	계란
allergen_milk	bool	N	우유
allergen_soy	bool	N	대두
allergen_peanut	bool	N	땅콩
allergen_nut	bool	N	견과류
allergen_fish	bool	N	생선
allergen_shellfish	bool	N	조개/갑각류
allergen_pork	bool	N	돼지고기
allergen_beef	bool	N	소고기
allergen_shrimp	bool	N	새우

4.2. AnalyzeResponse
Field	Type	Required	설명
assistant_text	string	Y	키오스크가 사용자에게 말해줄 멘트 (React TTS + UI)
actions	array<KioskAction>	Y	장바구니/옵션 변경을 위한 액션 리스트
should_finish	boolean	Y	true면 주문을 끝내고 결제 단계로 진행
next_scene	string	Y	다음 화면/상태

actions[].type:

"ADD_ITEM": 메뉴 추가

"REMOVE_ITEM": 메뉴 제거/수량 감소

"CUSTOMIZE": 재료/옵션 조정

"NONE": 장바구니 변경 없음 (안내/질문만)

actions[].customize:



{
  "add": ["케첩"],
  "remove": ["피클"]
}