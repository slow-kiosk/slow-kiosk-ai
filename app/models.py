from pydantic import BaseModel
from typing import List, Optional


class CartItem(BaseModel):
    menuId: str
    name: str
    qty: int
    price: float


class Cart(BaseModel):
    items: List[CartItem] = []


class MenuItem(BaseModel):
    menuId: str
    name: str
    category: Optional[str] = None
    price: float


class AnalyzeRequest(BaseModel):
    """Spring에서 넘어오는 요청 스키마.

    text  : 브라우저 STT로 인식된 사용자 말
    scene : 현재 화면 / 상태 (예: BURGER_SELECT, DRINK_SELECT, PAYMENT 등)
    cart  : 현재 장바구니 상태
    menu  : 현재 화면에서 선택 가능한 메뉴 목록
    """

    text: str
    scene: str
    cart: Cart
    menu: List[MenuItem]


class Action(BaseModel):
    """백엔드가 수행해야 할 액션.

    type   : ADD_ITEM / REMOVE_ITEM / CHANGE_QTY / GO_TO_PAYMENT / NONE 등
    menuId : 대상 메뉴 ID (없으면 None)
    qty    : 수량 변경이 필요한 경우 사용 (없으면 None)
    """

    type: str
    menuId: Optional[str] = None
    qty: Optional[int] = None


class AnalyzeResponse(BaseModel):
    """Python LLM 서버가 반환하는 응답 스키마."""

    assistant_text: str          # 사용자에게 보여줄/읽어줄 멘트
    actions: List[Action]        # 장바구니/화면 전환 액션
    should_finish: bool = False  # 결제 단계로 넘어갈지 여부