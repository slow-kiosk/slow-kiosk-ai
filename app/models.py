# app/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# 어떤 액션들을 지원할지 정의
ActionType = Literal["ADD_ITEM", "REMOVE_ITEM", "CUSTOMIZE", "NONE"]


class Customization(BaseModel):
    """
    메뉴 1개에 대해 '재료/옵션'을 어떻게 바꿀지 표현.
    예)
    add: ["케첩"]
    remove: ["피클", "양파"]
    """
    add: List[str] = Field(default_factory=list)
    remove: List[str] = Field(default_factory=list)


class KioskAction(BaseModel):
    """
    LLM이 한 번의 발화에 대해 여러 액션을 반환할 수 있도록 배열로 설계.
    type:
      - ADD_ITEM: 장바구니에 메뉴 추가
      - REMOVE_ITEM: 장바구니에서 메뉴 제거/수량 감소
      - CUSTOMIZE: 선택된 메뉴의 재료/옵션 수정
      - NONE: 실제 장바구니 변경은 없음 (안내/질문만)
    """
    type: ActionType
    menuId: Optional[str] = None
    qty: Optional[int] = 1
    customize: Optional[Customization] = None


class CartItem(BaseModel):
    menuId: str
    qty: int = 1


class Cart(BaseModel):
    items: List[CartItem] = Field(default_factory=list)


class MenuItem(BaseModel):
    """
    Spring/React에서 내려주는 메뉴 1개 스키마.
    CSV의 컬럼이랑 맞춰서 맞는 것만 쓰면 됨.
    """
    menuId: str
    name: str           # name_ko 사용해서 채우면 됨
    category: str
    price: int

    tags: List[str] = Field(default_factory=list)

    # 새로 추가된 재료 관련
    ingredients_ko: Optional[str] = None      # "참깨빵, 양상추, 양파, 피클, 소고기 패티, ..."
    customizable_ko: Optional[str] = None     # "피클, 양파, 소스, 치즈, 베이컨"


class AnalyzeRequest(BaseModel):
    """
    React → Spring → Python 으로 들어오는 Body 형식
    """
    text: str                     # 브라우저 STT로 인식한 텍스트
    scene: str                    # 현재 화면/상황(예: GREETING, SELECT_BURGER 등)
    cart: Cart                    # 현재 장바구니 상태
    menu: List[MenuItem]          # 현재 화면에서 선택 가능한 메뉴 리스트


class AnalyzeResponse(BaseModel):
    """
    Python → Spring 으로 나가는 응답 형식
    """
    assistant_text: str           # React에서 TTS로 읽어줄 멘트
    actions: List[KioskAction]    # 장바구니에 반영할 액션들
    should_finish: bool           # 주문을 끝낼지 여부
    next_scene: str               # 다음 화면/상태 (예: "CONFIRM", "GREETING" 등)
