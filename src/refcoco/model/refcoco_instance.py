from pydantic import BaseModel, Field


class RefCOCOInfo(BaseModel):
    description: str
    url: str
    version: str
    year: int
    contributor: str
    date_created: str


class RefCOCOImage(BaseModel):
    file_name: str
    coco_url: str
    height: int
    width: int
    date_captured: str
    flickr_url: str
    id_: int = Field(alias='id')
    license_: int = Field(alias='license')


class RefCOCOSegmentation(BaseModel):
    counts: list[int]
    size: list[int]


class RefCOCOAnnotation(BaseModel):
    image_id: int
    category_id: int
    segmentation: list[list[float]] | RefCOCOSegmentation
    area: float
    bbox: list[float]
    is_crowd: int = Field(alias='iscrowd')
    id_: int = Field(alias='id')


class RefCOCOLicense(BaseModel):
    url: str
    name: str
    id_: int = Field(alias='id')


class RefCOCOCategory(BaseModel):
    super_category: str = Field(alias='supercategory')
    name: str
    id_: int = Field(alias='id')


class RefCOCOInstance(BaseModel):
    info: RefCOCOInfo
    images: list[RefCOCOImage]
    annotations: list[RefCOCOAnnotation]
    licenses: list[RefCOCOLicense]
    categories: list[RefCOCOCategory]
