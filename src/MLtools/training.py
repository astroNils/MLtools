import copy
import imantics
import pandas as pd
import torch
import yaml
import detectron2.utils.comm as comm
import albumentations as A
import contextlib
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
import pycocotools.mask as mask_util

from collections import OrderedDict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg, CfgNode
from detectron2.data import detection_utils as utils
from detectron2.evaluation import DatasetEvaluator
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultPredictor, DefaultTrainer, launch
from detectron2.engine.hooks import HookBase
from skimage.draw import polygon2mask
from pathlib import Path
from detectron2.config.config import CfgNode as CN
from dataclasses import dataclass, field
from typing import Dict
from typing import Any, Union
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table
from detectron2.solver import build

try:
    from detectron2.evaluation.fast_eval_api import COCOeval_opt
except ImportError:
    COCOeval_opt = COCOeval

setup_logger()

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

# need to fix when several of the same name (for example of OneOf)
# fix it the same way as in augments_DataMapper.
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


def save_yaml(filepath: Union[str, Path], content: Any, width: int = 120):
    with open(filepath, "w") as f:
        yaml.dump(content, f, width=width)

def load_yaml(filepath: Union[str, Path]) -> Any:
    with open(filepath, "r") as f:
        content = yaml.full_load(f)
    return content

def load_aug_dict(filepath: Union[str, Path]) -> Any:
    """

    Args:
        filepath:
    Returns:

    A.from_dict(load_aug_dict(filepath))
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

def training_AlbumentMapper_polygon(config_file, config_file_complete, augmentation_file):

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:', torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:',
              torch.cuda.get_device_properties(0).total_memory / 1e9)

    device = "cuda" if use_cuda else "cpu"
    print("Device: ", device)

    class AlbumentationsMapper:
        """Mapper which uses `albumentations` augmentations"""

        def __init__(self, cfg, is_train: bool = True):
            aug_kwargs = cfg.aug_kwargs
            if is_train:
                aug_list = augments(aug_kwargs)
            else:
                aug_list = []
                # else, it gives an empty list, which is equivalent to NoOp
            self.transform = A.Compose(aug_list)
            self.is_train = is_train

            mode = "training" if is_train else "inference"
            print(
                f"[AlbumentationsMapper] Augmentations used in {mode}: {self.transform}")

        def __call__(self, dataset_dict):
            dataset_dict = copy.deepcopy(
                dataset_dict)  # it will be modified by code below
            image = utils.read_image(dataset_dict["file_name"], format="BGR")
            masks = batchpolygon2mask(dataset_dict["annotations"],
                                      dataset_dict["height"],
                                      dataset_dict["width"])

            # I could change it when it is is_train False --> to do nothing

            transformed = self.transform(image=image, masks=masks)
            transformed_image = transformed['image']
            transformed_masks = transformed['masks']

            # create empty dataframe
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


    """
    Original code from https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py
    Just modified to show AP@40
    """

    def boulder_summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets,
                              mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((18,))
            stats[0] = _summarize(1, maxDets=self.params.maxDets[2])
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small',
                                  maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium',
                                  maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large',
                                  maxDets=self.params.maxDets[2])
            stats[6] = _summarize(1, iouThr=.5, areaRng='small',
                                  maxDets=self.params.maxDets[2])
            stats[7] = _summarize(1, iouThr=.5, areaRng='medium',
                                  maxDets=self.params.maxDets[2])
            stats[8] = _summarize(1, iouThr=.5, areaRng='large',
                                  maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[10] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[11] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[12] = _summarize(0, areaRng='small',
                                   maxDets=self.params.maxDets[2])
            stats[13] = _summarize(0, areaRng='medium',
                                   maxDets=self.params.maxDets[2])
            stats[14] = _summarize(0, areaRng='large',
                                   maxDets=self.params.maxDets[2])
            stats[15] = _summarize(0, iouThr=.5, areaRng='small',
                                   maxDets=self.params.maxDets[2])
            stats[16] = _summarize(0, iouThr=.5, areaRng='medium',
                                   maxDets=self.params.maxDets[2])
            stats[17] = _summarize(0, iouThr=.5, areaRng='large',
                                   maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    print("HACKING: overriding COCOeval.summarize = boulder_summarize...")
    COCOeval.summarize = boulder_summarize

    class BoulderEvaluator(DatasetEvaluator):
        """
        Evaluate AR for object proposals, AP for instance detection/segmentation, AP
        for keypoint detection outputs using COCO's metrics.
        See http://cocodataset.org/#detection-eval and
        http://cocodataset.org/#keypoints-eval to understand its metrics.

        In addition to COCO, this evaluator is able to support any bounding box detection,
        instance segmentation, or keypoint detection dataset.
        """

        def __init__(
                self,
                dataset_name,
                tasks=None,
                distributed=True,
                output_dir=None,
                *,
                max_dets_per_image=None,
                use_fast_impl=True,
                kpt_oks_sigmas=(),
        ):
            """
            Args:
                dataset_name (str): name of the dataset to be evaluated.
                    It must have either the following corresponding metadata:

                        "json_file": the path to the COCO format annotation

                    Or it must be in detectron2's standard dataset format
                    so it can be converted to COCO format automatically.
                tasks (tuple[str]): tasks that can be evaluated under the given
                    configuration. A task is one of "bbox", "segm", "keypoints".
                    By default, will infer this automatically from predictions.
                distributed (True): if True, will collect results from all ranks and run evaluation
                    in the main process.
                    Otherwise, will only evaluate the results in the current process.
                output_dir (str): optional, an output directory to dump all
                    results predicted on the dataset. The dump contains two files:

                    1. "instances_predictions.pth" a file in torch serialization
                       format that contains all the raw original predictions.
                    2. "coco_instances_results.json" a json file in COCO's result
                       format.
                use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                    Although the results should be very close to the official implementation in COCO
                    API, it is still recommended to compute results with the official API for use in
                    papers. The faster implementation also uses more RAM.
                kpt_oks_sigmas (list[float]): The sigmas used to calculate keypoint OKS.
                    See http://cocodataset.org/#keypoints-eval
                    When empty, it will use the defaults in COCO.
                    Otherwise it should be the same length as ROI_KEYPOINT_HEAD.NUM_KEYPOINTS.
            """
            self._logger = logging.getLogger(__name__)
            self._distributed = distributed
            self._output_dir = output_dir
            self._use_fast_impl = use_fast_impl

            if max_dets_per_image is None:
                max_dets_per_image = [1, 10, 100]
            else:
                max_dets_per_image = [1, 10, max_dets_per_image]
            self._max_dets_per_image = max_dets_per_image

            if tasks is not None and isinstance(tasks, CfgNode):
                kpt_oks_sigmas = (
                    tasks.TEST.KEYPOINT_OKS_SIGMAS if not kpt_oks_sigmas else kpt_oks_sigmas
                )
                self._logger.warn(
                    "COCO Evaluator instantiated using config, this is deprecated behavior."
                    " Please pass in explicit arguments instead."
                )
                self._tasks = None  # Infering it from predictions should be better
            else:
                self._tasks = tasks

            self._cpu_device = torch.device("cpu")

            self._metadata = MetadataCatalog.get(dataset_name)
            if not hasattr(self._metadata, "json_file"):
                self._logger.info(
                    f"'{dataset_name}' is not registered by `register_coco_instances`."
                    " Therefore trying to convert it to COCO format ..."
                )

                cache_path = os.path.join(output_dir,
                                          f"{dataset_name}_coco_format.json")
                self._metadata.json_file = cache_path
                convert_to_coco_json(dataset_name, cache_path)

            json_file = PathManager.get_local_path(self._metadata.json_file)
            with contextlib.redirect_stdout(io.StringIO()):
                self._coco_api = COCO(json_file)

            # Test set json files do not contain annotations (evaluation must be
            # performed using the COCO evaluation server).
            self._do_evaluation = "annotations" in self._coco_api.dataset
            if self._do_evaluation:
                self._kpt_oks_sigmas = kpt_oks_sigmas

        def reset(self):
            self._predictions = []

        def process(self, inputs, outputs):
            """
            Args:
                inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                    It is a list of dict. Each dict corresponds to an image and
                    contains keys like "height", "width", "file_name", "image_id".
                outputs: the outputs of a COCO model. It is a list of dicts with key
                    "instances" that contains :class:`Instances`.
            """
            for input, output in zip(inputs, outputs):
                prediction = {"image_id": input["image_id"]}

                if "instances" in output:
                    instances = output["instances"].to(self._cpu_device)
                    prediction["instances"] = instances_to_coco_json(instances,
                                                                     input[
                                                                         "image_id"])
                if "proposals" in output:
                    prediction["proposals"] = output["proposals"].to(
                        self._cpu_device)
                if len(prediction) > 1:
                    self._predictions.append(prediction)

        def evaluate(self, img_ids=None):
            """
            Args:
                img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
            """
            if self._distributed:
                comm.synchronize()
                predictions = comm.gather(self._predictions, dst=0)
                predictions = list(itertools.chain(*predictions))

                if not comm.is_main_process():
                    return {}
            else:
                predictions = self._predictions

            if len(predictions) == 0:
                self._logger.warning(
                    "[VinbigdataEvaluator] Did not receive valid predictions.")
                return {}

            if self._output_dir:
                PathManager.mkdirs(self._output_dir)
                file_path = os.path.join(self._output_dir,
                                         "instances_predictions.pth")
                with PathManager.open(file_path, "wb") as f:
                    torch.save(predictions, f)

            self._results = OrderedDict()
            if "proposals" in predictions[0]:
                self._eval_box_proposals(predictions)
            if "instances" in predictions[0]:
                self._eval_predictions(predictions, img_ids=img_ids)
            # Copy so the caller can do whatever with results
            return copy.deepcopy(self._results)

        def _tasks_from_predictions(self, predictions):
            """
            Get COCO API "tasks" (i.e. iou_type) from COCO-format predictions.
            """
            tasks = {"bbox"}
            for pred in predictions:
                if "segmentation" in pred:
                    tasks.add("segm")
                if "keypoints" in pred:
                    tasks.add("keypoints")
            return sorted(tasks)

        def _eval_predictions(self, predictions, img_ids=None):
            """
            Evaluate predictions. Fill self._results with the metrics of the tasks.
            """
            self._logger.info("Preparing results for COCO format ...")
            coco_results = list(
                itertools.chain(*[x["instances"] for x in predictions]))
            tasks = self._tasks or self._tasks_from_predictions(coco_results)

            # unmap the category ids for COCO
            if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
                dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
                all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
                num_classes = len(all_contiguous_ids)
                assert min(all_contiguous_ids) == 0 and max(
                    all_contiguous_ids) == num_classes - 1

                reverse_id_mapping = {v: k for k, v in
                                      dataset_id_to_contiguous_id.items()}
                for result in coco_results:
                    category_id = result["category_id"]
                    assert category_id < num_classes, (
                        f"A prediction has class={category_id}, "
                        f"but the dataset only has {num_classes} classes and "
                        f"predicted class id should be in [0, {num_classes - 1}]."
                    )
                    result["category_id"] = reverse_id_mapping[category_id]

            if self._output_dir:
                file_path = os.path.join(self._output_dir,
                                         "coco_instances_results.json")
                self._logger.info("Saving results to {}".format(file_path))
                with PathManager.open(file_path, "w") as f:
                    f.write(json.dumps(coco_results))
                    f.flush()

            if not self._do_evaluation:
                self._logger.info(
                    "Annotations are not available for evaluation.")
                return

            self._logger.info(
                "Evaluating predictions with {} COCO API...".format(
                    "unofficial" if self._use_fast_impl else "official"
                )
            )
            for task in sorted(tasks):
                coco_eval = (
                    _evaluate_predictions_on_coco(
                        self._coco_api,
                        coco_results,
                        task,
                        kpt_oks_sigmas=self._kpt_oks_sigmas,
                        use_fast_impl=self._use_fast_impl,
                        img_ids=img_ids,
                        max_dets_per_image=self._max_dets_per_image,
                    )
                    if len(coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )

                res = self._derive_coco_results(
                    coco_eval, task,
                    class_names=self._metadata.get("thing_classes")
                )
                self._results[task] = res

        def _eval_box_proposals(self, predictions):
            """
            Evaluate the box proposals in predictions.
            Fill self._results with the metrics for "box_proposals" task.
            """
            if self._output_dir:
                # Saving generated box proposals to file.
                # Predicted box_proposals are in XYXY_ABS mode.
                bbox_mode = BoxMode.XYXY_ABS.value
                ids, boxes, objectness_logits = [], [], []
                for prediction in predictions:
                    ids.append(prediction["image_id"])
                    boxes.append(
                        prediction["proposals"].proposal_boxes.tensor.numpy())
                    objectness_logits.append(
                        prediction["proposals"].objectness_logits.numpy())

                proposal_data = {
                    "boxes": boxes,
                    "objectness_logits": objectness_logits,
                    "ids": ids,
                    "bbox_mode": bbox_mode,
                }
                with PathManager.open(
                        os.path.join(self._output_dir, "box_proposals.pkl"),
                        "wb") as f:
                    pickle.dump(proposal_data, f)

            if not self._do_evaluation:
                self._logger.info(
                    "Annotations are not available for evaluation.")
                return

            self._logger.info("Evaluating bbox proposals ...")
            res = {}
            areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
            for limit in [100, 1000]:
                for area, suffix in areas.items():
                    stats = _evaluate_box_proposals(predictions, self._coco_api,
                                                    area=area, limit=limit)
                    key = "AR{}@{:d}".format(suffix, limit)
                    res[key] = float(stats["ar"].item() * 100)
            self._logger.info("Proposal metrics: \n" + create_small_table(res))
            self._results["box_proposals"] = res

        def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
            """
            Derive the desired score numbers from summarized COCOeval.

            Args:
                coco_eval (None or COCOEval): None represents no predictions from model.
                iou_type (str):
                class_names (None or list[str]): if provided, will use it to predict
                    per-category AP.

            Returns:
                a dict of {metric name: score}
            """

            metrics = {
                "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
                "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
                "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
            }[iou_type]

            if coco_eval is None:
                self._logger.warn("No predictions from the model!")
                return {metric: float("nan") for metric in metrics}

            # the standard metrics
            results = {
                metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[
                                                                idx] >= 0 else "nan")
                for idx, metric in enumerate(metrics)
            }
            self._logger.info(
                "Evaluation results for {}: \n".format(
                    iou_type) + create_small_table(results)
            )
            if not np.isfinite(sum(results.values())):
                self._logger.info(
                    "Some metrics cannot be computed and is shown as NaN.")

            if class_names is None or len(class_names) <= 1:
                return results
            # Compute per-category AP
            # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
            precisions = coco_eval.eval["precision"]
            # precision has dims (iou, recall, cls, area range, max dets)
            assert len(class_names) == precisions.shape[2]

            results_per_category = []
            for idx, name in enumerate(class_names):
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                precision = precisions[:, :, idx, 0, -1]
                precision = precision[precision > -1]
                ap = np.mean(precision) if precision.size else float("nan")
                results_per_category.append(
                    ("{}".format(name), float(ap * 100)))

            # tabulate it
            N_COLS = min(6, len(results_per_category) * 2)
            results_flatten = list(itertools.chain(*results_per_category))
            results_2d = itertools.zip_longest(
                *[results_flatten[i::N_COLS] for i in range(N_COLS)])
            table = tabulate(
                results_2d,
                tablefmt="pipe",
                floatfmt=".3f",
                headers=["category", "AP"] * (N_COLS // 2),
                numalign="left",
            )
            self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

            results.update(
                {"AP-" + name: ap for name, ap in results_per_category})
            return results

    def instances_to_coco_json(instances, img_id):
        """
        Dump an "Instances" object to a COCO-format json that's used for evaluation.

        Args:
            instances (Instances):
            img_id (int): the image id

        Returns:
            list[dict]: list of json annotations in COCO format.
        """
        num_instance = len(instances)
        if num_instance == 0:
            return []

        boxes = instances.pred_boxes.tensor.numpy()
        boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        boxes = boxes.tolist()
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()

        has_mask = instances.has("pred_masks")
        if has_mask:
            # use RLE to encode the masks, because they are too large and takes memory
            # since this evaluator stores outputs of the entire dataset
            rles = [
                mask_util.encode(
                    np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
                for mask in instances.pred_masks
            ]
            for rle in rles:
                # "counts" is an array encoded by mask_util as a byte-stream. Python3's
                # json writer which always produces strings cannot serialize a bytestream
                # unless you decode it. Thankfully, utf-8 works out (which is also what
                # the pycocotools/_mask.pyx does).
                rle["counts"] = rle["counts"].decode("utf-8")

        has_keypoints = instances.has("pred_keypoints")
        if has_keypoints:
            keypoints = instances.pred_keypoints

        results = []
        for k in range(num_instance):
            result = {
                "image_id": img_id,
                "category_id": classes[k],
                "bbox": boxes[k],
                "score": scores[k],
            }
            if has_mask:
                result["segmentation"] = rles[k]
            if has_keypoints:
                # In COCO annotations,
                # keypoints coordinates are pixel indices.
                # However our predictions are floating point coordinates.
                # Therefore we subtract 0.5 to be consistent with the annotation format.
                # This is the inverse of data loading logic in `datasets/coco.py`.
                keypoints[k][:, :2] -= 0.5
                result["keypoints"] = keypoints[k].flatten().tolist()
            results.append(result)
        return results

    # inspired from Detectron:
    # https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L255 # noqa
    def _evaluate_box_proposals(dataset_predictions, coco_api, thresholds=None,
                                area="all", limit=None):
        """
        Evaluate detection proposal recall metrics. This function is a much
        faster alternative to the official COCO API recall evaluation code. However,
        it produces slightly different results.
        """
        # Record max overlap value for each gt box
        # Return vector of overlap values
        areas = {
            "all": 0,
            "small": 1,
            "medium": 2,
            "large": 3,
            "96-128": 4,
            "128-256": 5,
            "256-512": 6,
            "512-inf": 7,
        }
        area_ranges = [
            [6 ** 2, 1e5 ** 2],  # all
            [6 ** 2, 16 ** 2],  # small
            # [16 ** 2, 32 ** 2],  # small
            [16 ** 2, 32 ** 2],  # medium
            [32 ** 2, 1e5 ** 2],  # large
            [96 ** 2, 128 ** 2],  # 96-128
            [128 ** 2, 256 ** 2],  # 128-256
            [256 ** 2, 512 ** 2],  # 256-512
            [512 ** 2, 1e5 ** 2],
        ]  # 512-inf
        assert area in areas, "Unknown area range: {}".format(area)
        area_range = area_ranges[areas[area]]
        gt_overlaps = []
        num_pos = 0

        for prediction_dict in dataset_predictions:
            predictions = prediction_dict["proposals"]

            # sort predictions in descending order
            # TODO maybe remove this and make it explicit in the documentation
            inds = predictions.objectness_logits.sort(descending=True)[1]
            predictions = predictions[inds]

            ann_ids = coco_api.getAnnIds(imgIds=prediction_dict["image_id"])
            anno = coco_api.loadAnns(ann_ids)
            gt_boxes = [
                BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                for obj in anno
                if obj["iscrowd"] == 0
            ]
            gt_boxes = torch.as_tensor(gt_boxes).reshape(-1,
                                                         4)  # guard against no boxes
            gt_boxes = Boxes(gt_boxes)
            gt_areas = torch.as_tensor(
                [obj["area"] for obj in anno if obj["iscrowd"] == 0])

            if len(gt_boxes) == 0 or len(predictions) == 0:
                continue

            valid_gt_inds = (gt_areas >= area_range[0]) & (
                    gt_areas <= area_range[1])
            gt_boxes = gt_boxes[valid_gt_inds]

            num_pos += len(gt_boxes)

            if len(gt_boxes) == 0:
                continue

            if limit is not None and len(predictions) > limit:
                predictions = predictions[:limit]

            overlaps = pairwise_iou(predictions.proposal_boxes, gt_boxes)

            _gt_overlaps = torch.zeros(len(gt_boxes))
            for j in range(min(len(predictions), len(gt_boxes))):
                # find which proposal box maximally covers each gt box
                # and get the iou amount of coverage for each gt box
                max_overlaps, argmax_overlaps = overlaps.max(dim=0)

                # find which gt box is 'best' covered (i.e. 'best' = most iou)
                gt_ovr, gt_ind = max_overlaps.max(dim=0)
                assert gt_ovr >= 0
                # find the proposal box that covers the best covered gt box
                box_ind = argmax_overlaps[gt_ind]
                # record the iou coverage of this gt box
                _gt_overlaps[j] = overlaps[box_ind, gt_ind]
                assert _gt_overlaps[j] == gt_ovr
                # mark the proposal box and the gt box as used
                overlaps[box_ind, :] = -1
                overlaps[:, gt_ind] = -1

            # append recorded iou coverage level
            gt_overlaps.append(_gt_overlaps)
        gt_overlaps = (
            torch.cat(gt_overlaps, dim=0) if len(gt_overlaps) else torch.zeros(
                0,
                dtype=torch.float32)
        )
        gt_overlaps, _ = torch.sort(gt_overlaps)

        if thresholds is None:
            step = 0.05
            thresholds = torch.arange(0.5, 0.95 + 1e-5, step,
                                      dtype=torch.float32)
            # thresholds = torch.arange(0.4, 0.95 + 1e-5, step, dtype=torch.float32)
        recalls = torch.zeros_like(thresholds)
        # compute recall for each iou threshold
        for i, t in enumerate(thresholds):
            recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
        # ar = 2 * np.trapz(recalls, thresholds)
        ar = recalls.mean()
        return {
            "ar": ar,
            "recalls": recalls,
            "thresholds": thresholds,
            "gt_overlaps": gt_overlaps,
            "num_pos": num_pos,
        }

    def _evaluate_predictions_on_coco(
            coco_gt, coco_results, iou_type, kpt_oks_sigmas=None,
            use_fast_impl=True, img_ids=None, max_dets_per_image=None
    ):
        """
        Evaluate the coco results using COCOEval API.
        """
        assert len(coco_results) > 0

        if iou_type == "segm":
            coco_results = copy.deepcopy(coco_results)
            # When evaluating mask AP, if the results contain bbox, cocoapi will
            # use the box area as the area of the instance, instead of the mask area.
            # This leads to a different definition of small/medium/large.
            # We remove the bbox field to let mask AP use mask area.
            for c in coco_results:
                c.pop("bbox", None)

        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = (COCOeval_opt if use_fast_impl else COCOeval)(coco_gt,
                                                                  coco_dt,
                                                                  iou_type)

        # HACKING: overwrite areaRng
        coco_eval.params.areaRng = [[6 ** 2, 1e5 ** 2], [6 ** 2, 16 ** 2],
                                    [16 ** 2, 32 ** 2], [32 ** 2, 1e5 ** 2]]
        # coco_eval.params.areaRng = [[6 ** 2, 1e5 ** 2], [16 ** 2, 32 ** 2], [16 ** 2, 32 ** 2], [32 ** 2, 1e5 ** 2]]

        # For COCO, the default max_dets_per_image is [1, 10, 100].
        if max_dets_per_image is None:
            max_dets_per_image = [1, 10, 100]  # Default from COCOEval
        else:
            assert (
                    len(max_dets_per_image) >= 3
            ), "COCOeval requires maxDets (and max_dets_per_image) to have length at least 3"
            # In the case that user supplies a custom input for max_dets_per_image,
            # apply COCOevalMaxDets to evaluate AP with the custom input.
            if max_dets_per_image[2] != 100:
                None
                # coco_eval = COCOevalMaxDets(coco_gt, coco_dt,
                #                            iou_type)  # this need to be changed

        if iou_type != "keypoints":
            coco_eval.params.maxDets = max_dets_per_image

        if img_ids is not None:
            coco_eval.params.imgIds = img_ids

        if iou_type == "keypoints":
            # Use the COCO default keypoint OKS sigmas unless overrides are specified
            if kpt_oks_sigmas:
                assert hasattr(coco_eval.params,
                               "kpt_oks_sigmas"), "pycocotools is too old!"
                coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)
            # COCOAPI requires every detection and every gt to have keypoints, so
            # we just take the first entry from both
            num_keypoints_dt = len(coco_results[0]["keypoints"]) // 3
            num_keypoints_gt = len(
                next(iter(coco_gt.anns.values()))["keypoints"]) // 3
            num_keypoints_oks = len(coco_eval.params.kpt_oks_sigmas)
            assert num_keypoints_oks == num_keypoints_dt == num_keypoints_gt, (
                f"[BoulderEvaluator] Prediction contain {num_keypoints_dt} keypoints. "
                f"Ground truth contains {num_keypoints_gt} keypoints. "
                f"The length of cfg.TEST.KEYPOINT_OKS_SIGMAS is {num_keypoints_oks}. "
                "They have to agree with each other. For meaning of OKS, please refer to "
                "http://cocodataset.org/#keypoints-eval."
            )

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval

    class MyTrainer(DefaultTrainer):
        @classmethod
        def build_train_loader(cls, cfg, is_train=True, sampler=None):
            return build_detection_train_loader(
                cfg, mapper=AlbumentationsMapper(cfg, is_train), sampler=sampler
            )

        @classmethod
        def build_test_loader(cls, cfg, dataset_name):
            return build_detection_test_loader(
                cfg, dataset_name, mapper=AlbumentationsMapper(cfg, False)
            )

        @classmethod
        def build_evaluator(cls, cfg, dataset_name):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
            os.makedirs(output_folder, exist_ok=True)
            return BoulderEvaluator(dataset_name, ("segm",), False, output_folder,
                                 max_dets_per_image=1000)

    class ValidationLoss(HookBase):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg.clone()
            self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST[0]
            self._loader = iter(MyTrainer.build_train_loader(self.cfg, is_train=False))  # False, for not applying any transforms

        def after_step(self):
            data = next(self._loader)
            with torch.no_grad():
                loss_dict = self.trainer.model(data)

                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {"val_" + k: v.item() for k, v in
                                     comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(
                    loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    self.trainer.storage.put_scalars(
                        total_val_loss=losses_reduced,
                        **loss_dict_reduced)

    @dataclass
    class BOULDERconfig:
        # augmentations
        aug_kwargs: Dict = field(default_factory=lambda: {})

        def update(self, param_dict: Dict) -> "BOULDERconfig":
            # Overwrite by `param_dict`
            for key, value in param_dict.items():
                if not hasattr(self, key):
                    raise ValueError(f"[ERROR] Unexpected key for flag = {key}")
                setattr(self, key, value)
            return self

    # read augmentations
    augmentations_dict = load_yaml(augmentation_file) # change here
    flags = BOULDERconfig().update(augmentations_dict)
    cfg = get_cfg()
    cfg.merge_from_file(config_file)

    # Save complete config file
    with open(config_file_complete, "w") as f:
        f.write(cfg.dump())

    cfg.aug_kwargs = CN(flags.aug_kwargs)  # pass aug_kwargs to cfg
    cfg.MODEL.DEVICE = device

    # training
    trainer = MyTrainer(cfg)
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])
    # swap the order of PeriodicWriter and ValidationLoss
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    trainer.train()

def training_AlbumentMapper_mask(config_file, config_file_complete, augmentation_file, min_area_npixels, optimizer_name):

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:', torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:',
              torch.cuda.get_device_properties(0).total_memory / 1e9)

    device = "cuda" if use_cuda else "cpu"
    print("Device: ", device)


    # be careful bbox numpy use + 1, which still needs to be tested

    class AlbumentationsMapper:
        """Mapper which uses `albumentations` augmentations"""

        def __init__(self, cfg, is_train: bool = True):
            aug_kwargs = cfg.aug_kwargs
            if is_train:
                aug_dict = aug_kwargs
                self.transform = A.from_dict(aug_dict)
            else:
                self.transform = A.Compose([])

            self.is_train = is_train

            mode = "training" if is_train else "inference"
            print(
                f"[AlbumentationsMapper] Augmentations used in {mode}: {self.transform}")

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

            idx = np.where(np.array(n_px_per_mask) >= cfg.min_area_npixels)[0].tolist()
            transformed_masks_final = [transformed_masks_filtered[i] for i in
                                       range(len(transformed_masks_filtered)) if
                                       i in idx]
            transformed_bboxes = [bbox_numpy(i) for i in
                                  transformed_masks_final]

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


    """
    Original code from https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py
    Just modified to show AP@40
    """

    def boulder_summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets,
                              mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((18,))
            stats[0] = _summarize(1, maxDets=self.params.maxDets[2])
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small',
                                  maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium',
                                  maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large',
                                  maxDets=self.params.maxDets[2])
            stats[6] = _summarize(1, iouThr=.5, areaRng='small',
                                  maxDets=self.params.maxDets[2])
            stats[7] = _summarize(1, iouThr=.5, areaRng='medium',
                                  maxDets=self.params.maxDets[2])
            stats[8] = _summarize(1, iouThr=.5, areaRng='large',
                                  maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[10] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[11] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[12] = _summarize(0, areaRng='small',
                                   maxDets=self.params.maxDets[2])
            stats[13] = _summarize(0, areaRng='medium',
                                   maxDets=self.params.maxDets[2])
            stats[14] = _summarize(0, areaRng='large',
                                   maxDets=self.params.maxDets[2])
            stats[15] = _summarize(0, iouThr=.5, areaRng='small',
                                   maxDets=self.params.maxDets[2])
            stats[16] = _summarize(0, iouThr=.5, areaRng='medium',
                                   maxDets=self.params.maxDets[2])
            stats[17] = _summarize(0, iouThr=.5, areaRng='large',
                                   maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    print("HACKING: overriding COCOeval.summarize = boulder_summarize...")
    COCOeval.summarize = boulder_summarize

    class BoulderEvaluator(DatasetEvaluator):
        """
        Evaluate AR for object proposals, AP for instance detection/segmentation, AP
        for keypoint detection outputs using COCO's metrics.
        See http://cocodataset.org/#detection-eval and
        http://cocodataset.org/#keypoints-eval to understand its metrics.

        In addition to COCO, this evaluator is able to support any bounding box detection,
        instance segmentation, or keypoint detection dataset.
        """

        def __init__(
                self,
                dataset_name,
                tasks=None,
                distributed=True,
                output_dir=None,
                *,
                max_dets_per_image=None,
                use_fast_impl=True,
                kpt_oks_sigmas=(),
        ):
            """
            Args:
                dataset_name (str): name of the dataset to be evaluated.
                    It must have either the following corresponding metadata:

                        "json_file": the path to the COCO format annotation

                    Or it must be in detectron2's standard dataset format
                    so it can be converted to COCO format automatically.
                tasks (tuple[str]): tasks that can be evaluated under the given
                    configuration. A task is one of "bbox", "segm", "keypoints".
                    By default, will infer this automatically from predictions.
                distributed (True): if True, will collect results from all ranks and run evaluation
                    in the main process.
                    Otherwise, will only evaluate the results in the current process.
                output_dir (str): optional, an output directory to dump all
                    results predicted on the dataset. The dump contains two files:

                    1. "instances_predictions.pth" a file in torch serialization
                       format that contains all the raw original predictions.
                    2. "coco_instances_results.json" a json file in COCO's result
                       format.
                use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                    Although the results should be very close to the official implementation in COCO
                    API, it is still recommended to compute results with the official API for use in
                    papers. The faster implementation also uses more RAM.
                kpt_oks_sigmas (list[float]): The sigmas used to calculate keypoint OKS.
                    See http://cocodataset.org/#keypoints-eval
                    When empty, it will use the defaults in COCO.
                    Otherwise it should be the same length as ROI_KEYPOINT_HEAD.NUM_KEYPOINTS.
            """
            self._logger = logging.getLogger(__name__)
            self._distributed = distributed
            self._output_dir = output_dir
            self._use_fast_impl = use_fast_impl

            if max_dets_per_image is None:
                max_dets_per_image = [1, 10, 100]
            else:
                max_dets_per_image = [1, 10, max_dets_per_image]
            self._max_dets_per_image = max_dets_per_image

            if tasks is not None and isinstance(tasks, CfgNode):
                kpt_oks_sigmas = (
                    tasks.TEST.KEYPOINT_OKS_SIGMAS if not kpt_oks_sigmas else kpt_oks_sigmas
                )
                self._logger.warn(
                    "COCO Evaluator instantiated using config, this is deprecated behavior."
                    " Please pass in explicit arguments instead."
                )
                self._tasks = None  # Infering it from predictions should be better
            else:
                self._tasks = tasks

            self._cpu_device = torch.device("cpu")

            self._metadata = MetadataCatalog.get(dataset_name)
            if not hasattr(self._metadata, "json_file"):
                self._logger.info(
                    f"'{dataset_name}' is not registered by `register_coco_instances`."
                    " Therefore trying to convert it to COCO format ..."
                )

                cache_path = os.path.join(output_dir,
                                          f"{dataset_name}_coco_format.json")
                self._metadata.json_file = cache_path
                convert_to_coco_json(dataset_name, cache_path)

            json_file = PathManager.get_local_path(self._metadata.json_file)
            with contextlib.redirect_stdout(io.StringIO()):
                self._coco_api = COCO(json_file)

            # Test set json files do not contain annotations (evaluation must be
            # performed using the COCO evaluation server).
            self._do_evaluation = "annotations" in self._coco_api.dataset
            if self._do_evaluation:
                self._kpt_oks_sigmas = kpt_oks_sigmas

        def reset(self):
            self._predictions = []

        def process(self, inputs, outputs):
            """
            Args:
                inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                    It is a list of dict. Each dict corresponds to an image and
                    contains keys like "height", "width", "file_name", "image_id".
                outputs: the outputs of a COCO model. It is a list of dicts with key
                    "instances" that contains :class:`Instances`.
            """
            for input, output in zip(inputs, outputs):
                prediction = {"image_id": input["image_id"]}

                if "instances" in output:
                    instances = output["instances"].to(self._cpu_device)
                    prediction["instances"] = instances_to_coco_json(instances,
                                                                     input[
                                                                         "image_id"])
                if "proposals" in output:
                    prediction["proposals"] = output["proposals"].to(
                        self._cpu_device)
                if len(prediction) > 1:
                    self._predictions.append(prediction)

        def evaluate(self, img_ids=None):
            """
            Args:
                img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
            """
            if self._distributed:
                comm.synchronize()
                predictions = comm.gather(self._predictions, dst=0)
                predictions = list(itertools.chain(*predictions))

                if not comm.is_main_process():
                    return {}
            else:
                predictions = self._predictions

            if len(predictions) == 0:
                self._logger.warning(
                    "[VinbigdataEvaluator] Did not receive valid predictions.")
                return {}

            if self._output_dir:
                PathManager.mkdirs(self._output_dir)
                file_path = os.path.join(self._output_dir,
                                         "instances_predictions.pth")
                with PathManager.open(file_path, "wb") as f:
                    torch.save(predictions, f)

            self._results = OrderedDict()
            if "proposals" in predictions[0]:
                self._eval_box_proposals(predictions)
            if "instances" in predictions[0]:
                self._eval_predictions(predictions, img_ids=img_ids)
            # Copy so the caller can do whatever with results
            return copy.deepcopy(self._results)

        def _tasks_from_predictions(self, predictions):
            """
            Get COCO API "tasks" (i.e. iou_type) from COCO-format predictions.
            """
            tasks = {"bbox"}
            for pred in predictions:
                if "segmentation" in pred:
                    tasks.add("segm")
                if "keypoints" in pred:
                    tasks.add("keypoints")
            return sorted(tasks)

        def _eval_predictions(self, predictions, img_ids=None):
            """
            Evaluate predictions. Fill self._results with the metrics of the tasks.
            """
            self._logger.info("Preparing results for COCO format ...")
            coco_results = list(
                itertools.chain(*[x["instances"] for x in predictions]))
            tasks = self._tasks or self._tasks_from_predictions(coco_results)

            # unmap the category ids for COCO
            if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
                dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
                all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
                num_classes = len(all_contiguous_ids)
                assert min(all_contiguous_ids) == 0 and max(
                    all_contiguous_ids) == num_classes - 1

                reverse_id_mapping = {v: k for k, v in
                                      dataset_id_to_contiguous_id.items()}
                for result in coco_results:
                    category_id = result["category_id"]
                    assert category_id < num_classes, (
                        f"A prediction has class={category_id}, "
                        f"but the dataset only has {num_classes} classes and "
                        f"predicted class id should be in [0, {num_classes - 1}]."
                    )
                    result["category_id"] = reverse_id_mapping[category_id]

            if self._output_dir:
                file_path = os.path.join(self._output_dir,
                                         "coco_instances_results.json")
                self._logger.info("Saving results to {}".format(file_path))
                with PathManager.open(file_path, "w") as f:
                    f.write(json.dumps(coco_results))
                    f.flush()

            if not self._do_evaluation:
                self._logger.info(
                    "Annotations are not available for evaluation.")
                return

            self._logger.info(
                "Evaluating predictions with {} COCO API...".format(
                    "unofficial" if self._use_fast_impl else "official"
                )
            )
            for task in sorted(tasks):
                coco_eval = (
                    _evaluate_predictions_on_coco(
                        self._coco_api,
                        coco_results,
                        task,
                        kpt_oks_sigmas=self._kpt_oks_sigmas,
                        use_fast_impl=self._use_fast_impl,
                        img_ids=img_ids,
                        max_dets_per_image=self._max_dets_per_image,
                    )
                    if len(coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )

                res = self._derive_coco_results(
                    coco_eval, task,
                    class_names=self._metadata.get("thing_classes")
                )
                self._results[task] = res

        def _eval_box_proposals(self, predictions):
            """
            Evaluate the box proposals in predictions.
            Fill self._results with the metrics for "box_proposals" task.
            """
            if self._output_dir:
                # Saving generated box proposals to file.
                # Predicted box_proposals are in XYXY_ABS mode.
                bbox_mode = BoxMode.XYXY_ABS.value
                ids, boxes, objectness_logits = [], [], []
                for prediction in predictions:
                    ids.append(prediction["image_id"])
                    boxes.append(
                        prediction["proposals"].proposal_boxes.tensor.numpy())
                    objectness_logits.append(
                        prediction["proposals"].objectness_logits.numpy())

                proposal_data = {
                    "boxes": boxes,
                    "objectness_logits": objectness_logits,
                    "ids": ids,
                    "bbox_mode": bbox_mode,
                }
                with PathManager.open(
                        os.path.join(self._output_dir, "box_proposals.pkl"),
                        "wb") as f:
                    pickle.dump(proposal_data, f)

            if not self._do_evaluation:
                self._logger.info(
                    "Annotations are not available for evaluation.")
                return

            self._logger.info("Evaluating bbox proposals ...")
            res = {}
            areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
            for limit in [100, 1000]:
                for area, suffix in areas.items():
                    stats = _evaluate_box_proposals(predictions, self._coco_api,
                                                    area=area, limit=limit)
                    key = "AR{}@{:d}".format(suffix, limit)
                    res[key] = float(stats["ar"].item() * 100)
            self._logger.info("Proposal metrics: \n" + create_small_table(res))
            self._results["box_proposals"] = res

        def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
            """
            Derive the desired score numbers from summarized COCOeval.

            Args:
                coco_eval (None or COCOEval): None represents no predictions from model.
                iou_type (str):
                class_names (None or list[str]): if provided, will use it to predict
                    per-category AP.

            Returns:
                a dict of {metric name: score}
            """

            metrics = {
                "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
                "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
                "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
            }[iou_type]

            if coco_eval is None:
                self._logger.warn("No predictions from the model!")
                return {metric: float("nan") for metric in metrics}

            # the standard metrics
            results = {
                metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[
                                                                idx] >= 0 else "nan")
                for idx, metric in enumerate(metrics)
            }
            self._logger.info(
                "Evaluation results for {}: \n".format(
                    iou_type) + create_small_table(results)
            )
            if not np.isfinite(sum(results.values())):
                self._logger.info(
                    "Some metrics cannot be computed and is shown as NaN.")

            if class_names is None or len(class_names) <= 1:
                return results
            # Compute per-category AP
            # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
            precisions = coco_eval.eval["precision"]
            # precision has dims (iou, recall, cls, area range, max dets)
            assert len(class_names) == precisions.shape[2]

            results_per_category = []
            for idx, name in enumerate(class_names):
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                precision = precisions[:, :, idx, 0, -1]
                precision = precision[precision > -1]
                ap = np.mean(precision) if precision.size else float("nan")
                results_per_category.append(
                    ("{}".format(name), float(ap * 100)))

            # tabulate it
            N_COLS = min(6, len(results_per_category) * 2)
            results_flatten = list(itertools.chain(*results_per_category))
            results_2d = itertools.zip_longest(
                *[results_flatten[i::N_COLS] for i in range(N_COLS)])
            table = tabulate(
                results_2d,
                tablefmt="pipe",
                floatfmt=".3f",
                headers=["category", "AP"] * (N_COLS // 2),
                numalign="left",
            )
            self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

            results.update(
                {"AP-" + name: ap for name, ap in results_per_category})
            return results

    def instances_to_coco_json(instances, img_id):
        """
        Dump an "Instances" object to a COCO-format json that's used for evaluation.

        Args:
            instances (Instances):
            img_id (int): the image id

        Returns:
            list[dict]: list of json annotations in COCO format.
        """
        num_instance = len(instances)
        if num_instance == 0:
            return []

        boxes = instances.pred_boxes.tensor.numpy()
        boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        boxes = boxes.tolist()
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()

        has_mask = instances.has("pred_masks")
        if has_mask:
            # use RLE to encode the masks, because they are too large and takes memory
            # since this evaluator stores outputs of the entire dataset
            rles = [
                mask_util.encode(
                    np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
                for mask in instances.pred_masks
            ]
            for rle in rles:
                # "counts" is an array encoded by mask_util as a byte-stream. Python3's
                # json writer which always produces strings cannot serialize a bytestream
                # unless you decode it. Thankfully, utf-8 works out (which is also what
                # the pycocotools/_mask.pyx does).
                rle["counts"] = rle["counts"].decode("utf-8")

        has_keypoints = instances.has("pred_keypoints")
        if has_keypoints:
            keypoints = instances.pred_keypoints

        results = []
        for k in range(num_instance):
            result = {
                "image_id": img_id,
                "category_id": classes[k],
                "bbox": boxes[k],
                "score": scores[k],
            }
            if has_mask:
                result["segmentation"] = rles[k]
            if has_keypoints:
                # In COCO annotations,
                # keypoints coordinates are pixel indices.
                # However our predictions are floating point coordinates.
                # Therefore we subtract 0.5 to be consistent with the annotation format.
                # This is the inverse of data loading logic in `datasets/coco.py`.
                keypoints[k][:, :2] -= 0.5
                result["keypoints"] = keypoints[k].flatten().tolist()
            results.append(result)
        return results

    # inspired from Detectron:
    # https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L255 # noqa
    def _evaluate_box_proposals(dataset_predictions, coco_api, thresholds=None,
                                area="all", limit=None):
        """
        Evaluate detection proposal recall metrics. This function is a much
        faster alternative to the official COCO API recall evaluation code. However,
        it produces slightly different results.
        """
        # Record max overlap value for each gt box
        # Return vector of overlap values
        areas = {
            "all": 0,
            "small": 1,
            "medium": 2,
            "large": 3,
            "96-128": 4,
            "128-256": 5,
            "256-512": 6,
            "512-inf": 7,
        }
        area_ranges = [
            [6 ** 2, 1e5 ** 2],  # all
            [6 ** 2, 16 ** 2],  # small
            # [16 ** 2, 32 ** 2],  # small
            [16 ** 2, 32 ** 2],  # medium
            [32 ** 2, 1e5 ** 2],  # large
            [96 ** 2, 128 ** 2],  # 96-128
            [128 ** 2, 256 ** 2],  # 128-256
            [256 ** 2, 512 ** 2],  # 256-512
            [512 ** 2, 1e5 ** 2],
        ]  # 512-inf
        assert area in areas, "Unknown area range: {}".format(area)
        area_range = area_ranges[areas[area]]
        gt_overlaps = []
        num_pos = 0

        for prediction_dict in dataset_predictions:
            predictions = prediction_dict["proposals"]

            # sort predictions in descending order
            # TODO maybe remove this and make it explicit in the documentation
            inds = predictions.objectness_logits.sort(descending=True)[1]
            predictions = predictions[inds]

            ann_ids = coco_api.getAnnIds(imgIds=prediction_dict["image_id"])
            anno = coco_api.loadAnns(ann_ids)
            gt_boxes = [
                BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                for obj in anno
                if obj["iscrowd"] == 0
            ]
            gt_boxes = torch.as_tensor(gt_boxes).reshape(-1,
                                                         4)  # guard against no boxes
            gt_boxes = Boxes(gt_boxes)
            gt_areas = torch.as_tensor(
                [obj["area"] for obj in anno if obj["iscrowd"] == 0])

            if len(gt_boxes) == 0 or len(predictions) == 0:
                continue

            valid_gt_inds = (gt_areas >= area_range[0]) & (
                    gt_areas <= area_range[1])
            gt_boxes = gt_boxes[valid_gt_inds]

            num_pos += len(gt_boxes)

            if len(gt_boxes) == 0:
                continue

            if limit is not None and len(predictions) > limit:
                predictions = predictions[:limit]

            overlaps = pairwise_iou(predictions.proposal_boxes, gt_boxes)

            _gt_overlaps = torch.zeros(len(gt_boxes))
            for j in range(min(len(predictions), len(gt_boxes))):
                # find which proposal box maximally covers each gt box
                # and get the iou amount of coverage for each gt box
                max_overlaps, argmax_overlaps = overlaps.max(dim=0)

                # find which gt box is 'best' covered (i.e. 'best' = most iou)
                gt_ovr, gt_ind = max_overlaps.max(dim=0)
                assert gt_ovr >= 0
                # find the proposal box that covers the best covered gt box
                box_ind = argmax_overlaps[gt_ind]
                # record the iou coverage of this gt box
                _gt_overlaps[j] = overlaps[box_ind, gt_ind]
                assert _gt_overlaps[j] == gt_ovr
                # mark the proposal box and the gt box as used
                overlaps[box_ind, :] = -1
                overlaps[:, gt_ind] = -1

            # append recorded iou coverage level
            gt_overlaps.append(_gt_overlaps)
        gt_overlaps = (
            torch.cat(gt_overlaps, dim=0) if len(gt_overlaps) else torch.zeros(
                0,
                dtype=torch.float32)
        )
        gt_overlaps, _ = torch.sort(gt_overlaps)

        if thresholds is None:
            step = 0.05
            thresholds = torch.arange(0.5, 0.95 + 1e-5, step,
                                      dtype=torch.float32)
            # thresholds = torch.arange(0.4, 0.95 + 1e-5, step, dtype=torch.float32)
        recalls = torch.zeros_like(thresholds)
        # compute recall for each iou threshold
        for i, t in enumerate(thresholds):
            recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
        # ar = 2 * np.trapz(recalls, thresholds)
        ar = recalls.mean()
        return {
            "ar": ar,
            "recalls": recalls,
            "thresholds": thresholds,
            "gt_overlaps": gt_overlaps,
            "num_pos": num_pos,
        }

    def _evaluate_predictions_on_coco(
            coco_gt, coco_results, iou_type, kpt_oks_sigmas=None,
            use_fast_impl=True, img_ids=None, max_dets_per_image=None
    ):
        """
        Evaluate the coco results using COCOEval API.
        """
        assert len(coco_results) > 0

        if iou_type == "segm":
            coco_results = copy.deepcopy(coco_results)
            # When evaluating mask AP, if the results contain bbox, cocoapi will
            # use the box area as the area of the instance, instead of the mask area.
            # This leads to a different definition of small/medium/large.
            # We remove the bbox field to let mask AP use mask area.
            for c in coco_results:
                c.pop("bbox", None)

        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = (COCOeval_opt if use_fast_impl else COCOeval)(coco_gt,
                                                                  coco_dt,
                                                                  iou_type)

        # HACKING: overwrite areaRng
        coco_eval.params.areaRng = [[6 ** 2, 1e5 ** 2], [6 ** 2, 16 ** 2],
                                    [16 ** 2, 32 ** 2], [32 ** 2, 1e5 ** 2]]
        # coco_eval.params.areaRng = [[6 ** 2, 1e5 ** 2], [16 ** 2, 32 ** 2], [16 ** 2, 32 ** 2], [32 ** 2, 1e5 ** 2]]

        # For COCO, the default max_dets_per_image is [1, 10, 100].
        if max_dets_per_image is None:
            max_dets_per_image = [1, 10, 100]  # Default from COCOEval
        else:
            assert (
                    len(max_dets_per_image) >= 3
            ), "COCOeval requires maxDets (and max_dets_per_image) to have length at least 3"
            # In the case that user supplies a custom input for max_dets_per_image,
            # apply COCOevalMaxDets to evaluate AP with the custom input.
            if max_dets_per_image[2] != 100:
                None
                # coco_eval = COCOevalMaxDets(coco_gt, coco_dt,
                #                            iou_type)  # this need to be changed

        if iou_type != "keypoints":
            coco_eval.params.maxDets = max_dets_per_image

        if img_ids is not None:
            coco_eval.params.imgIds = img_ids

        if iou_type == "keypoints":
            # Use the COCO default keypoint OKS sigmas unless overrides are specified
            if kpt_oks_sigmas:
                assert hasattr(coco_eval.params,
                               "kpt_oks_sigmas"), "pycocotools is too old!"
                coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)
            # COCOAPI requires every detection and every gt to have keypoints, so
            # we just take the first entry from both
            num_keypoints_dt = len(coco_results[0]["keypoints"]) // 3
            num_keypoints_gt = len(
                next(iter(coco_gt.anns.values()))["keypoints"]) // 3
            num_keypoints_oks = len(coco_eval.params.kpt_oks_sigmas)
            assert num_keypoints_oks == num_keypoints_dt == num_keypoints_gt, (
                f"[BoulderEvaluator] Prediction contain {num_keypoints_dt} keypoints. "
                f"Ground truth contains {num_keypoints_gt} keypoints. "
                f"The length of cfg.TEST.KEYPOINT_OKS_SIGMAS is {num_keypoints_oks}. "
                "They have to agree with each other. For meaning of OKS, please refer to "
                "http://cocodataset.org/#keypoints-eval."
            )

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval

    class MyTrainer(DefaultTrainer):

        @classmethod
        def build_train_loader(cls, cfg, is_train=True, sampler=None):
            return build_detection_train_loader(
                cfg, mapper=AlbumentationsMapper(cfg, is_train), sampler=sampler
            )

        @classmethod
        def build_test_loader(cls, cfg, dataset_name):
            return build_detection_test_loader(
                cfg, dataset_name, mapper=AlbumentationsMapper(cfg, False)
            )

        @classmethod
        def build_evaluator(cls, cfg, dataset_name):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
            os.makedirs(output_folder, exist_ok=True)
            return BoulderEvaluator(dataset_name, ("segm",), False, output_folder,
                                 max_dets_per_image=1000)

        @classmethod
        def build_lr_scheduler(cls, cfg, optimizer):
            """
            It now calls :func:`detectron2.solver.build_lr_scheduler`.
            Overwrite it if you'd like a different scheduler.
            """
            return build.build_lr_scheduler(cfg, optimizer)

        @classmethod
        def build_optimizer(cls, cfg, model):
            """
            Build an optimizer from config.
            """
            params = build.get_default_optimizer_params(
                model,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            )

            optimizer_type = cfg.OPTIMIZER
            if optimizer_type == "SGD":
                return build.maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                    params,
                    cfg.SOLVER.BASE_LR,
                    momentum=cfg.SOLVER.MOMENTUM,
                    nesterov=cfg.SOLVER.NESTEROV,
                )
            elif optimizer_type == "ADAM":
                return build.maybe_add_gradient_clipping(cfg, torch.optim.Adam)(
                    params, cfg.SOLVER.BASE_LR)
            else:
                raise NotImplementedError(f"no optimizer type {optimizer_type}")

    class ValidationLoss(HookBase):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg.clone()
            self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST[0]
            self._loader = iter(MyTrainer.build_train_loader(self.cfg, is_train=False))  # False, for not applying any transforms

        def after_step(self):
            data = next(self._loader)
            with torch.no_grad():
                loss_dict = self.trainer.model(data)

                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {"val_" + k: v.item() for k, v in
                                     comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(
                    loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    self.trainer.storage.put_scalars(
                        total_val_loss=losses_reduced,
                        **loss_dict_reduced)

    @dataclass
    class BOULDERconfig:
        # augmentations
        aug_kwargs: Dict = field(default_factory=lambda: {})

        def update(self, param_dict: Dict) -> "BOULDERconfig":
            # Overwrite by `param_dict`
            for key, value in param_dict.items():
                if not hasattr(self, key):
                    raise ValueError(f"[ERROR] Unexpected key for flag = {key}")
                setattr(self, key, value)
            return self

    # read augmentations
    augmentations_dict = load_aug_dict(augmentation_file)
    flags = BOULDERconfig().update({"aug_kwargs": augmentations_dict})
    cfg = get_cfg()
    cfg.merge_from_file(config_file)

    # Save complete config file
    with open(config_file_complete, "w") as f:
        f.write(cfg.dump())

    cfg.aug_kwargs = CN(flags.aug_kwargs)
    cfg.min_area_npixels = min_area_npixels
    cfg.OPTIMIZER = optimizer_name
    cfg.MODEL.DEVICE = device

    # training
    trainer = MyTrainer(cfg)
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])
    # swap the order of PeriodicWriter and ValidationLoss
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    trainer.train()