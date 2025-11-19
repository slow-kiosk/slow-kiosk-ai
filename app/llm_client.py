import json
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from .models import AnalyzeRequest, AnalyzeResponse, Action

# .env 로컬 개발 편의를 위한 로드 (실제 서버 환경에서는 환경변수로 주입해도 됨)
load_dotenv()

client = OpenAI()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def build_prompt(req: AnalyzeRequest) -> str:
    """LLM에게 넘길 프롬프트를 생성합니다.

    - 장바구니/메뉴 요약
    - 현재 scene, 사용자 발화(text)를 함께 전달
    - JSON 형식으로만 응답하도록 강하게 요구
    """

    if req.cart.items:
        cart_summary_lines = [
            f"- {item.name}({item.menuId}): {item.qty}개, {int(item.price)}원"
            for item in req.cart.items
        ]
        cart_summary = "\n".join(cart_summary_lines)
    else:
        cart_summary = "없음"

    menu_summary_lines = [
        f"- {m.name}({m.menuId}): {int(m.price)}원, category={m.category or 'UNKNOWN'}"
        for m in req.menu
    ]
    menu_summary = "\n".join(menu_summary_lines) or "없음"

    prompt = f"""
    너는 패스트푸드 키오스크의 'AI 두뇌' 역할을 한다.

    입력으로:
    1) 사용자 발화(text)
    2) 현재 화면/상태(scene)
    3) 현재 장바구니(cart)
    4) 주문 가능한 메뉴 목록(menu)
    을 받는다.

    너의 목표:
    - 사용자의 의도를 이해하고,
    - 어떤 액션을 할지 JSON으로 반환한다.
    - 예: 메뉴 추가, 삭제, 수량 변경, 결제 화면으로 이동 등.

    반드시 아래 JSON 형식만 출력해라.
    절대로 설명 문장, 코드 블럭, 다른 텍스트를 섞지 마라.

    {{
      "assistant_text": "사용자에게 한국어로 보여줄/읽어줄 한 문장",
      "actions": [
        {{
          "type": "ADD_ITEM" | "REMOVE_ITEM" | "CHANGE_QTY" | "GO_TO_PAYMENT" | "NONE",
          "menuId": "메뉴 ID 또는 null",
          "qty": 정수 또는 null
        }}
      ],
      "should_finish": true 또는 false
    }}

    [현재 화면(scene)]
    {req.scene}

    [현재 장바구니(cart)]
    {cart_summary}

    [메뉴 목록(menu)]
    {menu_summary}

    [사용자 발화(text)]
    "{req.text}"
    """

    return prompt


def analyze_with_llm(req: AnalyzeRequest) -> AnalyzeResponse:
    """OpenAI LLM을 호출해서 AnalyzeResponse를 생성합니다."""

    prompt = build_prompt(req)

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "너는 JSON만 반환하는 키오스크 주문 엔진이다. "
                    "항상 유효한 JSON만 출력해야 한다."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    raw = completion.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # LLM이 형식을 살짝 어겼을 때 대비한 안전장치
        data = {
            "assistant_text": "죄송해요, 다시 한 번 천천히 말씀해 주세요.",
            "actions": [
                {"type": "NONE", "menuId": None, "qty": None},
            ],
            "should_finish": False,
        }

    # 누락 필드를 보완하기 위해 기본값을 채움
    if "actions" not in data or not isinstance(data["actions"], list):
        data["actions"] = [{"type": "NONE", "menuId": None, "qty": None}]
    normalized_actions = []
    for a in data["actions"]:
        a_type = a.get("type", "NONE")
        menu_id = a.get("menuId")
        qty = a.get("qty")
        normalized_actions.append(Action(type=a_type, menuId=menu_id, qty=qty))

    assistant_text = data.get(
        "assistant_text", "죄송해요, 다시 한 번 천천히 말씀해 주세요."
    )
    should_finish = bool(data.get("should_finish", False))

    return AnalyzeResponse(
        assistant_text=assistant_text,
        actions=normalized_actions,
        should_finish=should_finish,
    )