# app/models.py
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

# --------------------------------------
# 액션 타입 정의
# --------------------------------------
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


# --------------------------------------
# 장바구니 관련
# --------------------------------------
class CartItem(BaseModel):
    menuId: str
    qty: int = 1


class Cart(BaseModel):
    items: List[CartItem] = Field(default_factory=list)


# --------------------------------------
# 메뉴 정보 (CSV 컬럼 매핑)
# --------------------------------------
class MenuItem(BaseModel):
    """
    Spring/React에서 내려주는 메뉴 1개 스키마.
    CSV의 컬럼이랑 맞춰서 맞는 것만 쓰면 됨.
    """
    menuId: str
    name: str           # name_ko 사용해서 채우면 됨
    category: str
    price: int

    # 태그: "대표메뉴", "가성비", "매운맛", "맵지않음", "아이추천", "어르신추천", ...
    tags: List[str] = Field(default_factory=list)

    # 재료/커스터마이즈
    ingredients_ko: Optional[str] = None      # "참깨빵, 양상추, 양파, 피클, 소고기 패티, ..."
    customizable_ko: Optional[str] = None     # "피클, 양파, 소스, 치즈, 베이컨"

    # 영양 정보
    kcal: Optional[float] = None
    protein_g: Optional[float] = None
    fat_g: Optional[float] = None
    carbs_g: Optional[float] = None
    sugars_g: Optional[float] = None
    sodium_mg: Optional[float] = None

    # 알레르기/경고
    allergens_ko: Optional[str] = None           # "밀, 우유, 계란, 대두 함유"
    allergy_warning_ko: Optional[str] = None     # "우유, 밀 알레르기 있는 분은 섭취에 주의하세요."

    # 한 줄 영양 요약
    nutrition_summary_ko: Optional[str] = None   # "단백질이 풍부하고, 칼로리는 중간 수준입니다."


# --------------------------------------
# 대화 히스토리 (프론트/백이 넘겨줌)
# --------------------------------------
class HistoryTurn(BaseModel):
    """
    이전 턴 대화 내용.
    - role: "user" (사용자 발화) / "assistant" (AI가 말한 문장)
    - content: 그때 실제로 보이거나 들려줬던 텍스트
    """
    role: Literal["user", "assistant"]
    content: str


# --------------------------------------
# /analyze 요청/응답 모델
# --------------------------------------
class AnalyzeRequest(BaseModel):
    """
    React → Spring → Python 으로 들어오는 Body 형식
    """
    text: str                     # 브라우저 STT로 인식한 텍스트
    scene: str                    # 현재 화면/상황(예: GREETING, SELECT_BURGER 등)
    cart: Cart                    # 현재 장바구니 상태
    menu: List[MenuItem]          # 현재 화면에서 선택 가능한 메뉴 리스트

    # 🔹 추가: 최근 대화 히스토리 (optional)
    history: List[HistoryTurn] = Field(
        default_factory=list,
        description="이전 user/assistant 발화 히스토리 (최신이 뒤에 오도록)"
    )


class AnalyzeResponse(BaseModel):
    """
    Python → Spring 으로 나가는 응답 형식
    """
    assistant_text: str           # React에서 TTS로 읽어줄 멘트
    actions: List[KioskAction]    # 장바구니에 반영할 액션들
    should_finish: bool           # 주문을 끝낼지 여부
    next_scene: str               # 다음 화면/상태 (예: "CONFIRM", "GREETING" 등)
