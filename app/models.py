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
    CSV의 컬럼이랑 맞춰서, 있는 것만 채워서 넘기면 됨.
    """
    menuId: str
    name: str           # name_ko 사용해서 채우면 됨
    category: str       # BURGER / SET / SIDE / DRINK / DESSERT 등
    price: int          # 원 단위 가격

    # 태그 (대표메뉴, 매운맛, 치킨, 가성비 등)
    tags: List[str] = Field(default_factory=list)

    # 재료/커스터마이즈 관련
    ingredients_ko: Optional[str] = None      # "참깨빵, 양상추, 양파, 피클, 소고기 패티, ..."
    customizable_ko: Optional[str] = None     # "피클, 양파, 소스, 치즈, 베이컨"

    # 🔹 영양 정보 (있으면 사용, 없으면 None)
    kcal: Optional[int] = None                # 칼로리(kcal)
    protein_g: Optional[float] = None         # 단백질(g)
    fat_g: Optional[float] = None             # 지방(g)
    saturated_fat_g: Optional[float] = None   # 포화지방(g)
    carbs_g: Optional[float] = None           # 탄수화물(g)
    sugars_g: Optional[float] = None          # 당류(g)
    sodium_mg: Optional[int] = None           # 나트륨(mg)

    # 🔹 알레르기 정보 (텍스트 + 플래그)
    allergens_ko: Optional[str] = None        # "밀, 대두, 우유, 계란, 소고기" 등
    allergens_en: Optional[str] = None        # "wheat, soy, milk, egg, beef"

    allergen_wheat: Optional[bool] = None
    allergen_egg: Optional[bool] = None
    allergen_milk: Optional[bool] = None
    allergen_soy: Optional[bool] = None
    allergen_peanut: Optional[bool] = None
    allergen_nut: Optional[bool] = None
    allergen_fish: Optional[bool] = None
    allergen_shellfish: Optional[bool] = None
    allergen_pork: Optional[bool] = None
    allergen_beef: Optional[bool] = None
    allergen_shrimp: Optional[bool] = None

    # 🔹 한 줄 요약
    nutrition_summary_ko: Optional[str] = None  # "1회 제공량 기준 ~kcal, 단백질 ~g ..." 등
    allergy_warning_ko: Optional[str] = None    # "밀, 우유, 계란 포함, 알레르기 주의" 등


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
