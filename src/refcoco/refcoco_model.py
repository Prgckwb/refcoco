import json
import pickle
from pathlib import Path

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
    iscrowd: int
    id_: int = Field(alias='id')


class RefCOCOLicense(BaseModel):
    url: str
    name: str
    id_: int = Field(alias='id')


class RefCOCOCategory(BaseModel):
    supercategory: str
    name: str
    id_: int = Field(alias='id')


class RefCOCOInstance(BaseModel):
    info: RefCOCOInfo
    images: list[RefCOCOImage]
    annotations: list[RefCOCOAnnotation]
    licenses: list[RefCOCOLicense]
    categories: list[RefCOCOCategory]


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


class RefCOCO:
    def __init__(
            self,
            data_root: str | Path,
            dataset_name: str = 'refcoco',
            split_by: str = 'unc',
            coco_year: int = 2014
    ):
        self.data_root = Path(data_root)
        self.dataset_name = dataset_name
        self.split_by = split_by

        refcoco_dir = self.data_root / dataset_name
        image_dir = self.data_root / f'train{coco_year}'

        if dataset_name in ['refcoco', 'refcoco+', 'refcocog']:
            pass
        elif dataset_name in ['refclef']:
            raise NotImplementedError
        else:
            raise ValueError(f'Invalid dataset name: {dataset_name}')

        # Referring Expression Data
        ref_path = refcoco_dir / f'refs({split_by}).p'
        with ref_path.open('rb') as f:
            refexp_data = pickle.load(f)
        self.refexp_data = [RefCOCOExpression.model_validate(d) for d in refexp_data]

        # RefCOCO Data
        with (refcoco_dir / 'instances.json').open('r') as f:
            instance_data = json.load(f)
        self.refcoco_data = RefCOCOInstance.model_validate(instance_data)
