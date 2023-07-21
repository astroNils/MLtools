import cv2
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio
import torch
import torchvision

from pathlib import Path
from PIL import Image
from rasterio import features

import detectron2
import sys
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from itertools import chain

from matplotlib.patches import Rectangle

def cv2_imshow(a):
    """
    A replacement for cv2.imshow() for use in Jupyter notebooks.
    Args:

    a : np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. shape
      (N, M, 3) is an NxM BGR color image. shape (N, M, 4) is an NxM BGRA color
      image.
    """
    a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    return (a)