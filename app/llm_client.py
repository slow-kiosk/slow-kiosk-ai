# app/llm_client.py
import os
import json
import logging
from typing import List, Set, Deque, Tuple
from collections import deque

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from .models import (
    AnalyzeRequest,
    AnalyzeResponse,
    MenuItem,
    KioskAction,
)

# 🔹 로거 설정
logger = logging.getLogger(__name__)

# 🔹 .env 로딩 (OPENAI_API_KEY, OPENAI_MODEL 등)
load_dotenv(override=True)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

client = OpenAI(api_key=api_key)

# 🔹 기본 사용할 모델
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# 🔹 최근 대화 히스토리 (user, assistant) 3턴까지 유지
#   - 키오스크 한 세션 동안만 유지된다고 가정
RECENT_TURNS: Deque[Tuple[str, str]] = deque(maxlen=3)

SYSTEM_PROMPT = """
너는 한국 패스트푸드점 '슬로우버거' 키오스크의 AI 주문 도우미다.

역할:
- 사용자의 음성 인식 결과 텍스트(text), 현재 화면(scene), 장바구니(cart), 메뉴 목록(menu)을 보고
  1) 어떤 말을 해줄지(assistant_text)
  2) 장바구니를 어떻게 바꿀지(actions)
  3) 주문을 끝낼지 여부(should_finish)
  4) 다음 화면(next_scene)
  를 JSON으로 결정한다.

중요 규칙:
- 반드시 JSON만 출력한다. 설명 문장, 마크다운, 코드블럭 없이 순수 JSON만.
- JSON 스키마는 다음과 같아야 한다.

{
  "assistant_text": "string",
  "actions": [
    {
      "type": "ADD_ITEM | REMOVE_ITEM | CUSTOMIZE | NONE",
      "menuId": "string or null",
      "qty": 1,
      "customize": {
        "add": ["string"],
        "remove": ["string"]
      }
    }
  ],
  "should_finish": false,
  "next_scene": "string"
}

설명:
- assistant_text: 고객에게 들려줄 자연스러운 한국어 문장.
- actions:
  - ADD_ITEM: 장바구니에 해당 menuId를 qty만큼 추가.
  - REMOVE_ITEM: 장바구니에서 해당 menuId를 qty만큼 제거(0 이하면 아이템 삭제).
  - CUSTOMIZE: 이미 선택된 메뉴에 대해, 재료/옵션을 조정.
  - NONE: 장바구니 변화 없음(안내, 질문만 하는 경우).
- customize:
  - add: ["케첩", "양파 추가"] 처럼 추가 요청 재료
  - remove: ["피클", "양파"] 처럼 빼달라는 재료
- should_finish:
  - true: "주문 완료하고 결제 단계로" 가야 함
  - false: 계속 주문 진행
- next_scene:
  - 예: "GREETING", "SELECT_BURGER", "CUSTOMIZE_BURGER", "SELECT_SIDE", "SELECT_DRINK", "CONFIRM" 등
  - 특별히 지정하기 어렵다면 현재 scene을 그대로 사용.

[주요 scene 흐름 규칙]

키오스크의 화면/단계는 크게 다음과 같이 가정한다:
- GREETING: 인사, 메뉴 설명/추천 단계
- SELECT_BURGER: 버거/세트 메뉴를 고르는 단계
- CUSTOMIZE_BURGER: 방금 선택한 버거나 세트의 야채/소스/치즈 등 커스터마이즈 단계
- SELECT_SIDE: 사이드 메뉴(감자튀김, 치킨너겟 등) 선택 단계
- SELECT_DRINK: 음료(콜라, 제로 콜라, 사이즈 등) 선택 단계
- CONFIRM: 주문 최종 확인 및 결제 직전 단계

scene에 따라 next_scene과 assistant_text를 다음과 같이 설계하라:

1) GREETING
- 사용자가 "추천해줘", "뭐가 맛있어"라고 하면:
  - menu 목록에서 대표 BURGER/SET 2~4개 정도 골라서 추천.
  - actions는 보통 NONE.
  - next_scene은 "SELECT_BURGER" 정도로 넘기는 것을 기본으로 한다.
- 사용자가 바로 "치즈버거 세트 주세요"처럼 구체적으로 주문하면:
  - 해당 버거/세트를 ADD_ITEM으로 장바구니에 담는다.
  - assistant_text에서 "세트 담아드렸고, 야채나 소스는 빼거나 추가하실 부분 없으신가요?"와 같이
    다음 단계(CUSTOMIZE_BURGER)로 자연스럽게 이어질 멘트를 만든다.
  - next_scene = "CUSTOMIZE_BURGER"로 넘긴다.

2) SELECT_BURGER
- 사용자가 특정 버거/세트를 주문하면:
  - 해당 menuId로 ADD_ITEM 액션을 만든다.
  - assistant_text에서 "야채나 소스를 빼거나 추가하실까요?"처럼 커스터마이즈를 유도한다.
  - next_scene = "CUSTOMIZE_BURGER".
- 사용자가 "다른 메뉴 없어?", "다른 버거 있어?"라고 하면:
  - menu 배열을 참고해 몇 가지를 소개하고, next_scene은 그대로 "SELECT_BURGER"를 유지할 수 있다.

3) CUSTOMIZE_BURGER
- 사용자가 "양상추 빼고 피클 많이", "케첩 추가", "양파 빼줘" 등 재료 관련 요청을 하면:
  - CUSTOMIZE 액션을 사용한다.
  - menuId는 방금 선택했거나, 장바구니에 있는 해당 버거/세트의 menuId를 사용하라.
  - customize.add / customize.remove 에 알맞게 문자열을 채운다.
  - 커스터마이즈가 어느 정도 끝났다면 assistant_text에서
    "이제 사이드 메뉴를 골라볼까요?"처럼 자연스럽게 사이드로 유도하고
    next_scene = "SELECT_SIDE"로 넘긴다.
- 사용자가 "그대로 주세요", "야채는 기본으로" 라고 하면:
  - 커스터마이즈 없이 next_scene = "SELECT_SIDE".

4) SELECT_SIDE
- 사용자가 "감자튀김", "치즈스틱 추가", "사이드는 필요 없어요"라고 하면:
  - 감자튀김/치즈스틱 등은 ADD_ITEM 액션으로 장바구니에 추가.
  - 사이드가 필요 없다고 하면 actions는 NONE.
  - assistant_text에서 "이제 음료를 골라주세요." 또는 "음료는 어떻게 하실까요?"라고 말하고
    next_scene = "SELECT_DRINK".

5) SELECT_DRINK
- 사용자가 "콜라", "제로 콜라", "콜라 라지로"라고 하면:
  - 해당 음료를 ADD_ITEM으로 담는다.
  - assistant_text에서 "주문 내용을 한 번 더 확인해드릴게요."로 마무리하고
    next_scene = "CONFIRM".
- 사용자가 "음료는 필요 없어요"라고 하면:
  - actions는 NONE 혹은 필요하다면 세트 구성에 맞게 처리.
  - next_scene = "CONFIRM".

6) CONFIRM
- 사용자가 "네, 결제할게요", "그대로 주세요"라고 하면:
  - should_finish = true 로 설정.
  - next_scene는 "CONFIRM"으로 유지하거나, 시스템 정의에 맞는 완료 상태를 사용.
- 사용자가 "버거 하나 더", "사이드 바꿔줘" 등 수정을 요청하면:
  - ADD_ITEM / REMOVE_ITEM / CUSTOMIZE를 적절히 사용해 장바구니를 수정한다.
  - 필요하다면 next_scene를 다시 "SELECT_BURGER"나 "SELECT_SIDE" 등으로 돌려보내
    수정 과정을 거칠 수 있게 한다.
  - 결제 의사가 명확하지 않다면 should_finish는 false로 둔다.

성분/영양/알레르기 응답 규칙:
- 사용자가 "성분", "재료", "알레르기", "영양", "칼로리", "당", "나트륨" 등을 물어보면,
  menu 항목의 ingredients_ko, kcal, protein_g, fat_g, carbs_g, sugars_g, sodium_mg,
  allergens_ko, allergy_warning_ko, nutrition_summary_ko를 우선적으로 참고해서 답변해라.
- CSV/데이터에 없는 항목은 임의로 지어내지 말고,
  "해당 메뉴의 자세한 영양 정보는 준비되어 있지 않습니다."처럼 솔직하게 말하라.
- 알레르기 관련 질문에는 가능하면 allergy_warning_ko 내용을 활용하여
  "밀, 우유, 계란, 대두를 함유하고 있어 관련 알레르기가 있으시면 섭취를 피하시는 것이 좋습니다."
  같은 주의 문장을 함께 포함하라.
- 여러 메뉴를 비교해달라고 하면 kcal, sugars_g, sodium_mg 등을 기반으로
  상대적으로 가벼운/무거운 메뉴를 설명하되, 어디까지나 안내용 설명임을 전제로 말하라.

[메뉴 태그(tags) 활용 가이드]

- 각 메뉴에는 tags 배열이 있을 수 있다. 예:
  - "대표메뉴"
  - "가성비"
  - "매운맛"
  - "맵지않음"
  - "아이추천"
  - "어르신추천"
  - "부드러운"
  - "가벼운"
  - "포만감"
- 사용자의 발화에서 다음과 같은 의도가 보이면, tags/영양 정보를 참고해 1~3개 정도 추천하라.
(이하 생략… 위에서 작성해둔 내용과 동일)
"""

# =========================
# 내부 Helper 함수들
# =========================

def _format_cart(req: AnalyzeRequest) -> str:
    """LLM에게 보여줄 장바구니 요약 문자열."""
    if not req.cart.items:
        return "현재 장바구니는 비어 있습니다."
    lines = []
    for ci in req.cart.items:
        name = next((m.name for m in req.menu if m.menuId == ci.menuId), ci.menuId)
        lines.append(f"- {name}({ci.menuId}) x {ci.qty}")
    return "\n".join(lines)


def _format_menu(menu: List[MenuItem], limit: int = 40) -> str:
    """LLM에게 보여줄 간단한 메뉴 요약 (최대 limit개)."""
    lines = []
    for m in menu[:limit]:
        parts = [f"[{m.menuId}] {m.name} / {m.category} / {m.price}원"]

        if getattr(m, "kcal", None) is not None:
            parts.append(f"{m.kcal}kcal")
        if getattr(m, "ingredients_ko", None):
            parts.append(f"재료: {m.ingredients_ko}")
        if getattr(m, "customizable_ko", None):
            parts.append(f"조절 가능: {m.customizable_ko}")
        if getattr(m, "allergens_ko", None):
            parts.append(f"알레르기: {m.allergens_ko}")
        if getattr(m, "nutrition_summary_ko", None):
            parts.append(f"영양요약: {m.nutrition_summary_ko}")
        if m.tags:
            parts.append("태그: " + ", ".join(m.tags))

        lines.append(" / ".join(parts))

    if len(menu) > limit:
        lines.append(f"... (총 {len(menu)}개 메뉴 중 {limit}개만 표시)")

    return "\n".join(lines)


def _build_history_block() -> str:
    """
    최근 3턴의 (user, assistant) 대화를 텍스트로 정리.
    LLM이 직전 맥락을 이해할 수 있도록 system/user 프롬프트에 포함.
    """
    if not RECENT_TURNS:
        return "최근 대화 기록 없음."

    lines = []
    for i, (user_text, assistant_text) in enumerate(RECENT_TURNS, start=1):
        lines.append(f"[턴 {i}]\n사용자: {user_text}\nAI: {assistant_text}")
    return "\n\n".join(lines)


def build_messages(req: AnalyzeRequest):
    """OpenAI ChatCompletion에 넘길 messages 구성 (히스토리 + 현재 발화)."""
    history_str = _build_history_block()
    cart_str = _format_cart(req)
    menu_str = _format_menu(req.menu)

    user_prompt = f"""
[최근 대화 히스토리]
{history_str}

[이번 사용자 발화]
{req.text}

[현재 화면(scene)]
{req.scene}

[현재 장바구니]
{cart_str}

[주문 가능 메뉴 목록]
{menu_str}

위 정보를 보고 JSON만 출력해라.
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _build_safe_fallback_response(req: AnalyzeRequest) -> AnalyzeResponse:
    """LLM 호출 실패 / 파싱 실패 등 예외 상황에서 사용할 안전한 기본 응답."""
    return AnalyzeResponse(
        assistant_text="죄송합니다, 잠시 오류가 발생했어요. 다시 한 번만 말씀해 주시겠어요?",
        actions=[KioskAction(type="NONE", menuId=None, qty=1, customize=None)],
        should_finish=False,
        next_scene=req.scene,
    )


def _normalize_actions(raw_actions, valid_menu_ids: Set[str], current_scene: str):
    """
    LLM이 반환한 actions 리스트를 검증/보정한다.
    - type이 이상하면 NONE으로
    - menuId가 유효하지 않은데 ADD/REMOVE/CUSTOMIZE면 NONE으로 다운그레이드
    - menuId는 숫자/문자 상관없이 문자열로 통일해서 비교
    """
    default_action = {"type": "NONE", "menuId": None, "qty": 1, "customize": None}

    if not isinstance(raw_actions, list) or len(raw_actions) == 0:
        return [default_action]

    valid_types = {"ADD_ITEM", "REMOVE_ITEM", "CUSTOMIZE", "NONE"}
    fixed_actions = []

    for a in raw_actions:
        if not isinstance(a, dict):
            fixed_actions.append(default_action)
            continue

        t = a.get("type")
        if t not in valid_types:
            t = "NONE"

        # 🔹 menuId를 무조건 문자열로 변환
        raw_menu_id = a.get("menuId")
        menu_id = str(raw_menu_id) if raw_menu_id is not None else None

        qty = a.get("qty", 1)
        customize = a.get("customize")

        if t in {"ADD_ITEM", "REMOVE_ITEM", "CUSTOMIZE"}:
            if menu_id not in valid_menu_ids:
                # logger.warning(f"Invalid Menu ID filtered: {menu_id} (raw: {raw_menu_id})")
                fixed_actions.append(default_action)
                continue

        fixed_actions.append(
            {
                "type": t,
                "menuId": menu_id if t != "NONE" else None,
                "qty": qty,
                "customize": customize,
            }
        )

    return fixed_actions


# =========================
# 외부에 노출되는 주요 함수
# =========================

def call_llm(req: AnalyzeRequest) -> AnalyzeResponse:
    """
    /analyze 엔드포인트에서 사용하는 핵심 LLM 호출 함수.
    - 프롬프트 생성 (히스토리 포함)
    - OpenAI 호출
    - JSON 파싱
    - actions 검증/보정
    - 예외/에러 시 안전한 fallback 응답
    - should_finish가 true이면 히스토리 초기화
    """
    messages = build_messages(req)
    logger.info(f"[AI-REQ] scene={req.scene}, text={req.text}")

    try:
        completion = client.chat.completions.create(
            model=DEFAULT_MODEL,
            response_format={"type": "json_object"},
            messages=messages,
            temperature=0.3,
            timeout=10,
        )
        content = completion.choices[0].message.content
        logger.debug(f"[AI-RAW] {content}")
    except OpenAIError as e:
        logger.error(f"[AI-ERROR] OpenAIError: {e}")
        return _build_safe_fallback_response(req)
    except Exception as e:
        logger.error(f"[AI-ERROR] Unexpected error: {e}")
        return _build_safe_fallback_response(req)

    # JSON 파싱
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.error("[AI-ERROR] JSON 디코딩 실패, fallback 응답 사용")
        return _build_safe_fallback_response(req)

    # 필수 필드 기본값 보정
    data.setdefault(
        "assistant_text",
        "죄송합니다. 다시 한 번만 말씀해 주시겠어요?",
    )
    data.setdefault("should_finish", False)
    data.setdefault("next_scene", req.scene)

    # actions 검증/보정
    raw_actions = data.get("actions")
    valid_menu_ids = {m.menuId for m in req.menu}
    data["actions"] = _normalize_actions(raw_actions, valid_menu_ids, req.scene)

    assistant_text = data.get("assistant_text")
    logger.info(f"[AI-RES] scene={req.scene}, assistant_text={assistant_text}")

    # 🔹 히스토리 업데이트 (이번 턴 기록)
    try:
        RECENT_TURNS.append((req.text, assistant_text))
    except Exception as e:
        logger.warning(f"[AI-HISTORY] update failed: {e}")

    # 🔹 주문 완료 시 히스토리 초기화 (고객 한 명 세션 끝났다고 가정)
    if data.get("should_finish"):
        RECENT_TURNS.clear()
        logger.info("[AI-HISTORY] cleared due to should_finish=True")

    return AnalyzeResponse(**data)
