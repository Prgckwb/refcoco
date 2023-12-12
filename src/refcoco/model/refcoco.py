import json
import pickle
from pathlib import Path

from refcoco_instance import RefCOCOInstance
from refcoco_refexp import RefCOCOExpression


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
