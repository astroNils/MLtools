import copy
import pandas as pd
import torch
import albumentations as A
import numpy as np
from detectron2.data import detection_utils as utils
from detectron2.config import configurable
from pathlib import Path
from typing import Any, Union, Dict

__all__ = ["AlbumentMapper", "load_aug_dict"]

class AlbumentMapper:

    """Mapper which uses `albumentations` augmentations. Boulder outlines are
    defined as X and Y coordinates."""

    #@configurable
    def __init__(self, cfg, is_train: bool = True):
        aug_dict = load_aug_dict(cfg.MODEL.AUGMENTATIONS.PATH)
        if is_train:
            # the dict need to be created with the parameters below
            # A.Compose(aug_list, bbox_params=A.BboxParams(format='coco', min_area=16, label_fields=["bbox_classes"]))
            self.transform = A.from_dict(aug_dict)
        else:
            self.transform = A.Compose([A.NoOp(p=1.0)], bbox_params=A.BboxParams(format='coco', min_area=8, label_fields=["bbox_classes"]))
            # else, it gives an empty list, which is equivalent to NoOp
        self.is_train = is_train

        mode = "training" if is_train else "inference"
        print(f"[AlbumentMapper] Augmentations used in {mode}: {self.transform}")

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        bbox = [i["bbox"] for i in dataset_dict["annotations"]]  # get boxes
        category_id = [i["category_id"] for i in dataset_dict["annotations"]]

        transformed = self.transform(image=image, bboxes=bbox, bbox_classes=category_id)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_category_id = transformed["bbox_classes"]

        # Updating annotation
        df = pd.DataFrame()
        df["bbox"] = transformed_bboxes
        df["bbox_mode"] = 1
        df["category_id"] = transformed_category_id
        annos = df.to_dict(orient='records')

        image_shape = transformed_image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(transformed_image.transpose(2, 0, 1).astype("float32"))
        instances = utils.annotations_to_instances(annos, image_shape) # needs to be there
        dataset_dict["instances"] = instances #utils.filter_empty_instances(instances) #  --> this is done already above
        return dataset_dict

def load_aug_dict(filepath: Union[str, Path]) -> Any:
    """
    to generate Albu pipeline --> A.from_dict(load_aug_dict(filepath))
    """
    return(pd.read_json(filepath).to_dict())

def bbox_numpy(img, add_one=True):
    # similar behavior as Detectron2 (with +1, make sense in QGIS)
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    if add_one:
        return [float(xmin), float(ymin), float(xmax + 1), float(ymax + 1)]
    else:
        return [float(xmin), float(ymin), float(xmax), float(ymax)]