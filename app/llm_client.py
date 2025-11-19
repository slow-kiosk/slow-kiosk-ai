# app/llm_client.py
import os
import json
import logging
from typing import List, Set

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from .models import (
    AnalyzeRequest,
    AnalyzeResponse,
    MenuItem,
    KioskAction,
)

# 🔹 로거 설정 (상위에서 설정하면 그걸 따라감)
logger = logging.getLogger(__name__)

# 🔹 .env 로딩 (OPENAI_API_KEY)
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

client = OpenAI(api_key=api_key)

# 🔹 기본 사용할 모델 (필요시 .env에서 OPENAI_MODEL=gpt-4.1 등으로 교체 가능)
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

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
  - 예: "GREETING", "SELECT_BURGER", "SELECT_SIDE", "CONFIRM" 등
  - 특별히 지정하기 어렵다면 현재 scene을 그대로 사용.

재료 정보 활용 방법:
- 각 메뉴에는 ingredients_ko (재료 목록), customizable_ko (조절 가능한 항목)가 있을 수 있다.
- 사용자가 "피클 빼줘", "양파 많이", "케첩 추가해줘" 라고 말하면,
  - 현재 선택된 메뉴나 방금 추가하려는 메뉴를 기준으로
  - CUSTOMIZE 액션을 만들어라.
    예:
    {
      "type": "CUSTOMIZE",
      "menuId": "B001",
      "qty": 1,
      "customize": {
        "add": ["케첩"],
        "remove": ["피클"]
      }
    }

대화 예시 (개념적, 실제 응답에는 포함하지 말 것):

사용자: "와퍼 세트 하나랑 콜라 제로로 주세요. 피클은 빼주세요."
-> assistant_text:
   "스테디 와퍼 세트 1개와 콜라 제로로 담아드리고, 와퍼에서 피클은 빼드릴게요. 다른 메뉴도 추가하시겠어요?"
-> actions:
[
  { "type": "ADD_ITEM", "menuId": "B001", "qty": 1, "customize": null },
  { "type": "CUSTOMIZE", "menuId": "B001", "qty": 1,
    "customize": { "add": [], "remove": ["피클"] }
  }
]
-> should_finish: false
-> next_scene: "SELECT_SIDE"

주의:
- menu 배열에 없는 menuId를 사용하면 안 된다.
- 사용자가 메뉴를 물어보면, menu 배열에서 인기 있거나 잘 팔릴만한 메뉴를 2~4개 정도 간단히 소개해라.
- 매운 음식/비건/치킨/세트 같은 조건이 나오면, menu의 category, tags, 재료를 참고해서 추천해라.
- 사용자의 의도가 애매하면, 바로 결제 끝내지 말고 한 번 더 확인 질문을 하라.
"""


# ======================================
# 내부 Helper 함수들
# ======================================

def _format_cart(req: AnalyzeRequest) -> str:
    """LLM에게 보여줄 장바구니 요약 문자열."""
    if not req.cart.items:
        return "현재 장바구니는 비어 있습니다."
    lines = []
    for ci in req.cart.items:
        # menuId로 name 찾기
        name = next((m.name for m in req.menu if m.menuId == ci.menuId), ci.menuId)
        lines.append(f"- {name}({ci.menuId}) x {ci.qty}")
    return "\n".join(lines)


def _format_menu(menu: List[MenuItem], limit: int = 40) -> str:
    """
    LLM에게 보여줄 간단한 메뉴 요약.
    너무 길어지지 않도록 최대 limit개까지만 보여줌.
    """
    lines = []
    for i, m in enumerate(menu[:limit]):
        parts = [f"[{m.menuId}] {m.name} / {m.category} / {m.price}원"]
        if m.ingredients_ko:
            parts.append(f"재료: {m.ingredients_ko}")
        if m.customizable_ko:
            parts.append(f"조절 가능: {m.customizable_ko}")
        if m.tags:
            parts.append("태그: " + ", ".join(m.tags))
        lines.append(" / ".join(parts))
    if len(menu) > limit:
        lines.append(f"... (총 {len(menu)}개 메뉴 중 {limit}개만 표시)")
    return "\n".join(lines)


def build_messages(req: AnalyzeRequest):
    """OpenAI ChatCompletion에 넘길 messages 구성."""
    cart_str = _format_cart(req)
    menu_str = _format_menu(req.menu)

    user_prompt = f"""
[사용자 발화]
{req.text}

[현재 화면(scene)]
{req.scene}

[현재 장바구니]
{cart_str}

[주문 가능 메뉴 목록]
{menu_str}

위 정보를 보고 JSON만 출력해라.
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return messages


def _build_safe_fallback_response(req: AnalyzeRequest) -> AnalyzeResponse:
    """
    LLM 호출 실패 / 파싱 실패 등 예외 상황에서 사용할 안전한 기본 응답.
    """
    return AnalyzeResponse(
        assistant_text="죄송합니다, 잠시 오류가 발생했어요. 다시 한 번만 말씀해 주시겠어요?",
        actions=[
            KioskAction(type="NONE", menuId=None, qty=1, customize=None)
        ],
        should_finish=False,
        next_scene=req.scene,
    )


def _normalize_actions(raw_actions, valid_menu_ids: Set[str], current_scene: str):
    """
    LLM이 반환한 actions 리스트를 검증/보정한다.
    - type이 이상하면 NONE으로
    - menuId가 유효하지 않은데 ADD/REMOVE/CUSTOMIZE면 NONE으로 다운그레이드
    """
    default_action = {"type": "NONE", "menuId": None, "qty": 1, "customize": None}

    # actions 기본값
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

        menu_id = a.get("menuId")
        qty = a.get("qty", 1)
        customize = a.get("customize")

        # menuId가 필요한 타입인데 유효한 ID가 아니면 NONE으로 다운그레이드
        if t in {"ADD_ITEM", "REMOVE_ITEM", "CUSTOMIZE"}:
            if menu_id not in valid_menu_ids:
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


# ======================================
# 외부에 노출되는 주요 함수
# ======================================

def call_llm(req: AnalyzeRequest) -> AnalyzeResponse:
    """
    /analyze 엔드포인트에서 사용하는 핵심 LLM 호출 함수.
    - 프롬프트 생성
    - OpenAI 호출
    - JSON 파싱
    - actions 검증/보정
    - 예외/에러 시 안전한 fallback 응답
    """

    messages = build_messages(req)
    logger.info(f"[AI-REQ] scene={req.scene}, text={req.text}")

    try:
        completion = client.chat.completions.create(
            model=DEFAULT_MODEL,
            response_format={"type": "json_object"},
            messages=messages,
            temperature=0.3,
            timeout=10,  # 초 단위, 필요시 조정
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

    logger.info(
        f"[AI-RES] scene={req.scene}, assistant_text={data.get('assistant_text')}"
    )

    # Pydantic 모델로 최종 검증
    return AnalyzeResponse(**data)
