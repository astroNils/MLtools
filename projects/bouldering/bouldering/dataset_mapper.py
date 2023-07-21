import copy
import imantics
import pandas as pd
import torch
import yaml
import albumentations as A
import numpy as np
import pycocotools.mask as mask_util
from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode
from detectron2.config import configurable
from skimage.draw import polygon2mask
from pathlib import Path
from typing import Any, Union, Dict

__all__ = ["AlbumentMapper_polygon", "AlbumentMapper_bitmask", "load_aug_dict"]

class AlbumentMapper_polygon:

    """Mapper which uses `albumentations` augmentations. Boulder outlines are
    defined as X and Y coordinates."""

    @configurable
    def __init__(self, cfg, is_train: bool = True):
        aug_dict = load_aug_dict(cfg.MODEL.AUGMENTATIONS.PATH)
        if is_train:
            self.transform = A.from_dict(aug_dict)
        else:
            self.transform = A.Compose([])
            # else, it gives an empty list, which is equivalent to NoOp
        self.is_train = is_train

        mode = "training" if is_train else "inference"
        print(f"[AlbumentMapper_polygon] Augmentations used in {mode}: {self.transform}")

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        masks = batchpolygon2mask(dataset_dict["annotations"],
                                  dataset_dict["height"],
                                  dataset_dict["width"])


        transformed = self.transform(image=image, masks=masks)
        transformed_image = transformed['image']
        transformed_masks = transformed['masks']

        df = pd.DataFrame()

        # generate new bounding box (could have used the self.transform but I prefer this way)
        transformed_bbox = []
        idx = []
        for i, mask in enumerate(transformed_masks):
            if np.nonzero(mask)[0].shape[0] == 0:
                None
            else:
                pos = np.nonzero(mask)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                transformed_bbox.append(
                    [xmin, ymin, xmax, ymax])  # XYXY_ABS
                idx.append(i)

        # equivalent to utils.filter_empty_instances
        transformed_masks = [transformed_masks[i] for i in idx]
        idx, polygons = masks2polygons(transformed_masks)  # some masks do not have enough coordinates to create a polygon
        transformed_bbox = [transformed_bbox[i] for i in idx]

        # Updating annotation
        df["iscrowd"] = [0] * len(transformed_bbox)
        df["bbox"] = transformed_bbox
        df["category_id"] = 0
        df["segmentation"] = polygons
        df["bbox_mode"] = BoxMode.XYXY_ABS
        annos = df.to_dict(orient='records')

        image_shape = transformed_image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(transformed_image.transpose(2, 0, 1).astype("float32"))
        instances = utils.annotations_to_instances(annos, image_shape, mask_format="polygon")  # needs to be there
        dataset_dict["instances"] = instances  # utils.filter_empty_instances(instances) #  --> this is done already above
        return dataset_dict


class AlbumentMapper_bitmask:
    """Mapper which uses `albumentations` augmentations. Boulder outlines are
    defined as masks."""

    def __init__(self, cfg, is_train: bool = True):
        aug_dict = load_aug_dict(cfg.MODEL.AUGMENTATIONS.PATH)
        if is_train:
            self.transform = A.from_dict(aug_dict)
        else:
            self.transform = A.Compose([])

        self.is_train = is_train
        self.min_area_npixels = cfg.INPUT.MIN_AREA_NPIXELS

        mode = "training" if is_train else "inference"
        print(f"[AlbumentMapper_bitmask] Augmentations used in {mode}: {self.transform}")

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        masks = [mask_util.decode(i["segmentation"]) for i in dataset_dict["annotations"]]

        transformed = self.transform(image=image, masks=masks)
        transformed_image = transformed['image']
        transformed_masks = transformed['masks']  # need to remove empty mask
        transformed_masks_filtered = [i for i in transformed_masks if np.any(i)]  # removing if empty

        # Counting the number of pixels in a mask
        # and removing if smaller than min_area_npixels (number of pixels)
        n_px_per_mask = []
        for i in transformed_masks_filtered:
            n_px_per_mask.append(len(np.nonzero(i)[0]))

        idx = np.where(np.array(n_px_per_mask) >= self.min_area_npixels)[0].tolist()
        transformed_masks_final = [transformed_masks_filtered[i] for i in
                                   range(len(transformed_masks_filtered)) if
                                   i in idx]
        transformed_bboxes = [bbox_numpy(i) for i in
                              transformed_masks_final] # +1 on the bbox

        transformed_rle_masks = []

        # rle_mask
        for m in transformed_masks_final:
            transformed_rle_masks.append(
                mask_util.encode(np.asarray(m, order="F")))

        # Updating annotation
        df = pd.DataFrame()
        df["bbox"] = transformed_bboxes
        df["bbox_mode"] = 0
        df["category_id"] = 0
        df["segmentation"] = transformed_rle_masks
        annos = df.to_dict(orient='records')

        image_shape = transformed_image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(transformed_image.transpose(2, 0, 1).astype("float32"))

        instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")  # needs to be there
        dataset_dict["instances"] = instances  # utils.filter_empty_instances(instances) #  --> this is done already above
        return dataset_dict

def augments(aug_kwargs):
    aug_list = []
    for key in aug_kwargs:
        if key.startswith("OneOf"):
            OneOf_list = []
            aug_oneOf = aug_kwargs[key].get("transforms")
            prob_oneOf = {'p':aug_kwargs[key].get("p")}
            OneOf_list.extend([getattr(A, name)(**kwargs) for name, kwargs in aug_oneOf.items()])
            aug_list.extend([A.OneOf(OneOf_list, **prob_oneOf)])
        else:
            kwargs = aug_kwargs[key]
            aug_list.extend([getattr(A, key)(**kwargs)])
    return aug_list

def batchpolygon2mask(anno, height, width):
    masks = []
    for i in range(len(anno)):
        segm = anno[i]["segmentation"][0]
        r = np.array(segm).reshape((-1,2))[:,1]
        c = np.array(segm).reshape((-1,2))[:,0]
        # very important to change type otherwise bool make some aug to crash
        mask = polygon2mask((height,width),np.column_stack([r,c])).astype('uint8')
        masks.append(mask)
    return masks

def masks2polygons(masks):
    polygons = []
    idx = []
    for i, mask in enumerate(masks):
        try:
            p = imantics.Mask(mask).polygons()
            # Cannot create a polygon from 4 coordinates, 2 pairs of x,y
            if len(list(p[0])) > 4:
                polygons.append([list(p[0])])
                idx.append(i)
        except:
            None
    return idx, polygons

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