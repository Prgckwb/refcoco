from pydantic import BaseModel


class RefCOCOSentence(BaseModel):
    tokens: list[str]
    raw: str
    sent_id: int
    sent: str


class RefCOCOExpression(BaseModel):
    sent_ids: list[int]
    file_name: str
    ann_id: int
    ref_id: int
    image_id: int
    split: str
    sentences: list[RefCOCOSentence]
    category_id: int
