from pathlib import Path
import re
import numpy as np
import pandas as pd
def dfAPs(trained_model_p, eval_period):
    trained_model_p = Path(trained_model_p)
    config_p = trained_model_p / "log.txt"

    with open(config_p, "r") as src:
        data = src.read()
        matches = re.findall(".*d2.evaluation.testing INFO: copypaste:.*", data)
        n = len(matches)
        idx = np.arange(2, n, 3)
        APs = [matches[i].split(": ")[-1] for i in idx]

    APs.insert(0, 'AP,AP50,AP75,APs,APm,APl')
    iteration = np.arange(0, (len(APs)-1)*eval_period, eval_period)
    APs_s = "\n".join(APs)
    AP_log_p = trained_model_p / "APs.txt"

    with open(AP_log_p, "w") as src:
        src.write(APs_s)

    df = pd.read_csv(AP_log_p)
    df["iteration"] = iteration
    return df

def dfAPs(trained_model_p, eval_period):
    trained_model_p = Path(trained_model_p)
    config_p = trained_model_p / "log.txt"

    with open(config_p, "r") as src:
        data = src.read()
        matches = re.findall(".*d2.evaluation.testing INFO: copypaste:.*", data)
        n = len(matches)
        idx = np.arange(2, n, 3)
        APs = [matches[i].split(": ")[-1] for i in idx]

    APs.insert(0, 'AP,AP50,AP75,APs,APm,APl')
    iteration = np.arange(0, (len(APs)-1)*eval_period, eval_period)
    APs_s = "\n".join(APs)
    AP_log_p = trained_model_p / "APs.txt"

    with open(AP_log_p, "w") as src:
        src.write(APs_s)

    df = pd.read_csv(AP_log_p)
    df["iteration"] = iteration
    return df


# def instance2semantic(mask, color):
#
#     """
#     This function convert instance segmentation labels to semantic segmentation
#     labels based on the specified color.
#
#     NB! There are lot of assumptions in this function:
#     - The mask is one band
#     - the color is given as tuple (e.g, (255,0,0) for yellow...
#
#     Need to be generalized for multiple inputs, colors, mask dimension and ++
#
#     :param mask:
#     :param color:
#     :return:
#     """
#     mask = Path(mask)
#     array = Image.open(mask)
#     array = np.array(array)
#     height, width = array.shape
#     new_array = np.zeros((height, width, 3)).astype('uint8')
#
#     # mask for values larger or equal to 1
#     mask_array = array >= 1
#
#     for i, c in enumerate(color):
#         new_array[:, :, i][mask_array] = c
#
#     output_filename = mask.with_name(mask.stem + "_semantic" + mask.suffix)
#     new_image = Image.fromarray(new_array)
#     new_image.save(output_filename)
#
# def get_boxes_from_masks_XYXY(masks):
#     """ Helper, gets bounding boxes from masks. They seem a bit off, not sure why.. """
#     bboxes = []
#     updated_masks = []
#     for i, mask in enumerate(masks):
#         pos = np.nonzero(mask)
#         xmin = np.min(pos[1])  # for some reasons the bbox seems a bit off...
#         xmax = np.max(pos[1])
#         ymin = np.min(pos[0])
#         ymax = np.max(pos[0])
#
#         # let's extract the coordinates as expected in COCO format (the problem is that it cannot be converted back to a shapefile, because of the orders of tge
#         px = pos[1]
#         py = pos[0]
#         mask_coord = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
#         mask_coord = [p for x in mask_coord for p in x]
#
#         # few entries have only one pixel
#         # let's avoid those entries
#         if ((xmin == xmax) or (ymin == ymax)):
#             None
#         else:
#             bboxes.append([xmin, ymin, xmax, ymax])
#             updated_masks.append(
#                 mask_coord)  # only add masks that have a correct bounding box
#
#     num_masks = len(bboxes)
#
#     return bboxes, updated_masks, num_masks
#
# def toCOCO_datasetcatalog(root):
#
#     try:
#         root = root.as_posix()
#     except:
#         None
#
#     imgs_f = list(sorted(glob.glob(os.path.join(root, "images") + "/*image.png")))
#     masks_f = list(sorted(glob.glob(os.path.join(root, "masks") + "/*.png")))
#
#     n = len(imgs_f)
#
#     dataset_dicts = []
#
#     for i in range(n):
#
#         record = {}
#
#         img_path = Path(imgs_f[i])
#         mask_path = Path(masks_f[i])
#
#         # reading image
#         image = Image.open(img_path)
#         image = np.array(image)
#         (height, width) = image.shape
#
#         # reading mask
#         mask = Image.open(mask_path)
#         mask = np.array(mask).astype('uint16')  # (height,width)
#
#         # instances are encoded as different colors
#         obj_ids = np.unique(mask)
#
#         # first id is the background, so remove it
#         obj_ids = obj_ids[1:]
#
#         # split the color-encoded mask into a set
#         # of binary masks
#
#         masks = []
#         for obj_id in obj_ids:
#             masks.append((mask == obj_id) + 0)
#
#         bboxes, updated_masks, num_masks = get_boxes_from_masks_XYXY(masks)
#
#         #record["file_name"] = img_path.name # use relative path
#         record["file_name"] = img_path.with_name(
#             img_path.stem + "_fakergb" + img_path.suffix).as_posix()
#         record["image_id"] = i
#         record["height"] = height
#         record["width"] = width
#
#         objs = []
#         for nm in range(num_masks):
#             obj = {"bbox": bboxes[nm], "bbox_mode": BoxMode.XYXY_ABS,
#                    "segmentation": [updated_masks[nm]], "category_id": 0}
#             objs.append(obj)
#
#         record["annotations"] = objs
#         dataset_dicts.append(record)
#
#     return (dataset_dicts)
#
#
# def toCOCO_register_instances(root):
#
#     try:
#         root = root.as_posix()
#     except:
#         None
#
#     imgs_f = list(sorted(glob.glob(os.path.join(root, "images") + "/*image.png")))
#     masks_f = list(sorted(glob.glob(os.path.join(root, "masks") + "/*.png")))
#
#     n = len(imgs_f)
#
#     dataset_dicts = []
#
#     for i in range(n):
#
#         record = {}
#
#         img_path = Path(imgs_f[i])
#         mask_path = Path(masks_f[i])
#
#         # reading image
#         image = Image.open(img_path)
#         image = np.array(image)
#         (height, width) = image.shape
#
#         # reading mask
#         mask = Image.open(mask_path)
#         mask = np.array(mask).astype('uint16')  # (height,width)
#
#         # instances are encoded as different colors
#         obj_ids = np.unique(mask)
#
#         # first id is the background, so remove it
#         obj_ids = obj_ids[1:]
#
#         # split the color-encoded mask into a set
#         # of binary masks
#
#         masks = []
#         for obj_id in obj_ids:
#             masks.append((mask == obj_id) + 0)
#
#         bboxes, updated_masks, num_masks = get_boxes_from_masks_XYXY(masks)
#
#         record["file_name"] = img_path.as_posix() #using one band png (full path)
#         #record["file_name"] = img_path.with_name(
#         #    img_path.stem + "_fakergb" + img_path.suffix).as_posix()
#         record["image_id"] = i
#         record["height"] = height
#         record["width"] = width
#
#         objs = []
#         for nm in range(num_masks):
#             obj = {"bbox": bboxes[nm], "bbox_mode": BoxMode.XYXY_ABS,
#                    "segmentation": [updated_masks[nm]], "category_id": 0}
#             objs.append(obj)
#
#         record["annotations"] = objs
#         dataset_dicts.append(record)
#
#     return (dataset_dicts)
#
# # TODO figure out where to save the data....
# def tiling_raster_from_dataframe(df, output_folder, block_width, block_height):
#     output_folder = Path(output_folder)
#     output_folder_tif = (output_folder / "tif")
#     output_folder_png1 = (output_folder / "png-1band")
#     output_folder_png3 = (output_folder / "png-3band")
#     output_folder_tif.mkdir(parents=True, exist_ok=True)
#     output_folder_png1.mkdir(parents=True, exist_ok=True)
#     output_folder_png3.mkdir(parents=True, exist_ok=True)
#
#     for index, row in tqdm(df.iterrows(), total=df.shape[0]):
#
#         # this is only useful within the loop if generating tiling on multiple images
#         in_raster = row.raster_ap
#         src_profile = raster.get_raster_profile(in_raster)
#         win_profile = src_profile
#         win_profile["width"] = block_width
#         win_profile["height"] = block_height
#
#         arr = raster.read_raster(in_raster=in_raster,
#                                  bbox=rio.windows.Window(*row.rwindows))
#
#         # edge cases (in the East, and South, the extent can be beigger than the actual raster)
#         # read_raster will then return an array with not the dimension
#         h, w = arr.squeeze().shape
#
#         if (h, w) != (block_height, block_width):
#             arr = np.pad(arr.squeeze(),
#                          [(0, block_height - h), (0, block_width - w)],
#                          mode='constant', constant_values=0)
#             arr = np.expand_dims(arr, axis=0)
#
#         filename_tif = (
#                     output_folder_tif / row.file_name.replace(".png", ".tif"))
#         filename_png1 = (output_folder_tif / row.file_name)
#         filename_png3 = (output_folder_tif / row.NAC_id + "_image_3band.png")
#         win_profile["transform"] = Affine(*row["transform"])
#
#         raster.save_raster(filename_tif, arr, win_profile, is_image=False)  # generate tif
#         raster.tiff_to_png(filename_tif, filename_png1)  # generate png (1-band) # tif to png
#         raster.fake_RGB(filename_png1, filename_png3)  # generate png (3-band) # png to fakepng
#
#
# def split_global(df_bbox, split):
#     """
#     Global shuffling
#     """
#
#     np.random.seed(seed=27)
#     n = df_bbox.shape[0]
#     idx_shuffle = np.random.permutation(n)
#
#     training_idx, remaining_idx = np.split(idx_shuffle,
#                                            [int(split[0] * len(idx_shuffle))])
#     split_val = split[1] / (1 - split[0])  # split compare to remaining data
#     val_idx, test_idx = np.split(remaining_idx,
#                                  [int(split_val * len(remaining_idx))])
#
#     df_bbox["dataset"] = "train"
#     df_bbox["dataset"].iloc[val_idx] = "validation"
#     df_bbox["dataset"].iloc[test_idx] = "test"
#
#     return (df_bbox)
#
#
# def folder_structure(dataset_directory):
#     dataset_directory = Path(dataset_directory)
#     folders = ["train", "validation", "test"]
#     sub_folders = ["images", "labels"]
#
#     for f in folders:
#         for s in sub_folders:
#             new_folder = dataset_directory / f / s
#             Path(new_folder).mkdir(parents=True, exist_ok=True)
#
#
# def augments_DataMapper(aug_kwargs):
#     aug_list = []
#     for key in aug_kwargs:
#         if "_" in key:
#             kwargs = aug_kwargs[key]
#             aug_list.extend([getattr(T, key.split("_")[0])(**kwargs)])
#         elif key in ["RandomContrast", "RandomBrightness"]:
#             kwargs = aug_kwargs[key]
#             aug_list.extend([T.RandomApply(getattr(T, key.split("_")[0])(**kwargs), prob=0.5)])
#         else:
#             kwargs = aug_kwargs[key]
#             aug_list.extend([getattr(T, key)(**kwargs)])
#     return aug_list
#
# def training_DatasetMapper(config_file, config_file_complete, augmentation_file):
#
#     use_cuda = torch.cuda.is_available()
#     if use_cuda:
#         print('__CUDNN VERSION:', torch.backends.cudnn.version())
#         print('__Number CUDA Devices:', torch.cuda.device_count())
#         print('__CUDA Device Name:', torch.cuda.get_device_name(0))
#         print('__CUDA Device Total Memory [GB]:',
#               torch.cuda.get_device_properties(0).total_memory / 1e9)
#
#     device = "cuda" if use_cuda else "cpu"
#     print("Device: ", device)
#
#     def build_augmentation(cfg, is_train):
#         """
#         Create a list of default :class:`Augmentation` from config.
#         Now it includes resizing and flipping.
#
#         Returns:
#             list[Augmentation]
#         """
#         if is_train:
#             augmentation = augments_DataMapper(cfg.aug_kwargs)
#         else:
#             augmentation = [
#                 T.NoOpTransform()]  # or should I have the resizing by default?
#             # augmentation = [ResizeShortestEdge(short_edge_length=[640, 672, 704, 736, 768, 800], max_size=1333, sample_style='choice')]
#         return augmentation
#
#     class DatasetMapper:
#         """
#         A callable which takes a dataset dict in Detectron2 Dataset format,
#         and map it into a format used by the model.
#
#         This is the default callable to be used to map your dataset dict into training data.
#         You may need to follow it to implement your own one for customized logic,
#         such as a different way to read or transform images.
#         See :doc:`/tutorials/data_loading` for details.
#
#         The callable currently does the following:
#
#         1. Read the image from "file_name"
#         2. Applies cropping/geometric transforms to the image and annotations
#         3. Prepare data and annotations to Tensor and :class:`Instances`
#         """
#
#         @configurable
#         def __init__(
#                 self,
#                 is_train: bool,
#                 *,
#                 augmentations: List[Union[T.Augmentation, T.Transform]],
#                 image_format: str,
#                 use_instance_mask: bool = False,
#                 use_keypoint: bool = False,
#                 instance_mask_format: str = "polygon",
#                 keypoint_hflip_indices: Optional[np.ndarray] = None,
#                 precomputed_proposal_topk: Optional[int] = None,
#                 recompute_boxes: bool = False,
#         ):
#             """
#             NOTE: this interface is experimental.
#
#             Args:
#                 is_train: whether it's used in training or inference
#                 augmentations: a list of augmentations or deterministic transforms to apply
#                 image_format: an image format supported by :func:`detection_utils.read_image`.
#                 use_instance_mask: whether to process instance segmentation annotations, if available
#                 use_keypoint: whether to process keypoint annotations if available
#                 instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
#                     masks into this format.
#                 keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
#                 precomputed_proposal_topk: if given, will load pre-computed
#                     proposals from dataset_dict and keep the top k proposals for each image.
#                 recompute_boxes: whether to overwrite bounding box annotations
#                     by computing tight bounding boxes from instance mask annotations.
#             """
#             if recompute_boxes:
#                 assert use_instance_mask, "recompute_boxes requires instance masks"
#             # fmt: off
#             self.is_train = is_train
#             self.augmentations = T.AugmentationList(augmentations)
#             self.image_format = image_format
#             self.use_instance_mask = use_instance_mask
#             self.instance_mask_format = instance_mask_format
#             self.use_keypoint = use_keypoint
#             self.keypoint_hflip_indices = keypoint_hflip_indices
#             self.proposal_topk = precomputed_proposal_topk
#             self.recompute_boxes = recompute_boxes
#             # fmt: on
#             logger = logging.getLogger(__name__)
#             mode = "training" if is_train else "inference"
#             logger.info(
#                 f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")
#
#         @classmethod
#         def from_config(cls, cfg, is_train: bool = True):
#             augs = build_augmentation(cfg, is_train)
#             recompute_boxes = cfg.MODEL.MASK_ON  # I think this is good anyway, to recompute boxes!
#             ret = {
#                 "is_train": is_train,
#                 "augmentations": augs,
#                 "image_format": cfg.INPUT.FORMAT,
#                 "use_instance_mask": cfg.MODEL.MASK_ON,
#                 "instance_mask_format": cfg.INPUT.MASK_FORMAT,
#                 "use_keypoint": cfg.MODEL.KEYPOINT_ON,
#                 "recompute_boxes": recompute_boxes,
#             }
#
#             if cfg.MODEL.KEYPOINT_ON:
#                 ret[
#                     "keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(
#                     cfg.DATASETS.TRAIN)
#
#             if cfg.MODEL.LOAD_PROPOSALS:
#                 ret["precomputed_proposal_topk"] = (
#                     cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
#                     if is_train
#                     else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
#                 )
#             return ret
#
#         def _transform_annotations(self, dataset_dict, transforms, image_shape):
#             # USER: Modify this if you want to keep them for some reason.
#             for anno in dataset_dict["annotations"]:
#                 if not self.use_instance_mask:
#                     anno.pop("segmentation", None)
#                 if not self.use_keypoint:
#                     anno.pop("keypoints", None)
#
#             # USER: Implement additional transformations if you have other types of data
#             annos = [
#                 utils.transform_instance_annotations(
#                     obj, transforms, image_shape,
#                     keypoint_hflip_indices=self.keypoint_hflip_indices
#                 )
#                 for obj in dataset_dict.pop("annotations")
#                 if obj.get("iscrowd", 0) == 0
#             ]
#             instances = utils.annotations_to_instances(
#                 annos, image_shape, mask_format=self.instance_mask_format
#             )
#
#             # After transforms such as cropping are applied, the bounding box may no longer
#             # tightly bound the object. As an example, imagine a triangle object
#             # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
#             # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
#             # the intersection of original bounding box and the cropping box.
#             if self.recompute_boxes:
#                 instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
#             dataset_dict["instances"] = utils.filter_empty_instances(instances)
#
#         def __call__(self, dataset_dict):
#             """
#             Args:
#                 dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
#
#             Returns:
#                 dict: a format that builtin models in detectron2 accept
#             """
#             dataset_dict = copy.deepcopy(
#                 dataset_dict)  # it will be modified by code below
#             # USER: Write your own image loading if it's not from a file
#             image = utils.read_image(dataset_dict["file_name"],
#                                      format=self.image_format)
#             utils.check_image_size(dataset_dict, image)
#
#             # USER: Remove if you don't do semantic/panoptic segmentation.
#             if "sem_seg_file_name" in dataset_dict:
#                 sem_seg_gt = utils.read_image(
#                     dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
#             else:
#                 sem_seg_gt = None
#
#             aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
#             transforms = self.augmentations(aug_input)
#             image, sem_seg_gt = aug_input.image, aug_input.sem_seg
#
#             image_shape = image.shape[:2]  # h, w
#             # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
#             # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
#             # Therefore it's important to use torch.Tensor.
#             dataset_dict["image"] = torch.as_tensor(
#                 np.ascontiguousarray(image.transpose(2, 0, 1)))
#             if sem_seg_gt is not None:
#                 dataset_dict["sem_seg"] = torch.as_tensor(
#                     sem_seg_gt.astype("long"))
#
#             # USER: Remove if you don't use pre-computed proposals.
#             # Most users would not need this feature.
#             if self.proposal_topk is not None:
#                 utils.transform_proposals(
#                     dataset_dict, image_shape, transforms,
#                     proposal_topk=self.proposal_topk
#                 )
#
#             # need to be commented for calculating validation loss (because annotations are needed)
#             # if not self.is_train:
#             #    # USER: Modify this if you want to keep them for some reason.
#             #    dataset_dict.pop("annotations", None)
#             #    dataset_dict.pop("sem_seg_file_name", None)
#             #    return dataset_dict
#
#             if "annotations" in dataset_dict:
#                 self._transform_annotations(dataset_dict, transforms,
#                                             image_shape)
#
#             return dataset_dict
#
#     """
#     Original code from https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py
#     Just modified to show AP@40
#     """
#
#     def boulder_summarize(self):
#         '''
#         Compute and display summary metrics for evaluation results.
#         Note this functin can *only* be applied on the default parameter setting
#         '''
#
#         def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
#             p = self.params
#             iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
#             titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
#             typeStr = '(AP)' if ap == 1 else '(AR)'
#             iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
#                 if iouThr is None else '{:0.2f}'.format(iouThr)
#
#             aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
#             mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
#             if ap == 1:
#                 # dimension of precision: [TxRxKxAxM]
#                 s = self.eval['precision']
#                 # IoU
#                 if iouThr is not None:
#                     t = np.where(iouThr == p.iouThrs)[0]
#                     s = s[t]
#                 s = s[:, :, :, aind, mind]
#             else:
#                 # dimension of recall: [TxKxAxM]
#                 s = self.eval['recall']
#                 if iouThr is not None:
#                     t = np.where(iouThr == p.iouThrs)[0]
#                     s = s[t]
#                 s = s[:, :, aind, mind]
#             if len(s[s > -1]) == 0:
#                 mean_s = -1
#             else:
#                 mean_s = np.mean(s[s > -1])
#             print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets,
#                               mean_s))
#             return mean_s
#
#         def _summarizeDets():
#             stats = np.zeros((18,))
#             stats[0] = _summarize(1, maxDets=self.params.maxDets[2])
#             stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
#             stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
#             stats[3] = _summarize(1, areaRng='small',
#                                   maxDets=self.params.maxDets[2])
#             stats[4] = _summarize(1, areaRng='medium',
#                                   maxDets=self.params.maxDets[2])
#             stats[5] = _summarize(1, areaRng='large',
#                                   maxDets=self.params.maxDets[2])
#             stats[6] = _summarize(1, iouThr=.5, areaRng='small',
#                                   maxDets=self.params.maxDets[2])
#             stats[7] = _summarize(1, iouThr=.5, areaRng='medium',
#                                   maxDets=self.params.maxDets[2])
#             stats[8] = _summarize(1, iouThr=.5, areaRng='large',
#                                   maxDets=self.params.maxDets[2])
#             stats[9] = _summarize(0, maxDets=self.params.maxDets[0])
#             stats[10] = _summarize(0, maxDets=self.params.maxDets[1])
#             stats[11] = _summarize(0, maxDets=self.params.maxDets[2])
#             stats[12] = _summarize(0, areaRng='small',
#                                    maxDets=self.params.maxDets[2])
#             stats[13] = _summarize(0, areaRng='medium',
#                                    maxDets=self.params.maxDets[2])
#             stats[14] = _summarize(0, areaRng='large',
#                                    maxDets=self.params.maxDets[2])
#             stats[15] = _summarize(0, iouThr=.5, areaRng='small',
#                                    maxDets=self.params.maxDets[2])
#             stats[16] = _summarize(0, iouThr=.5, areaRng='medium',
#                                    maxDets=self.params.maxDets[2])
#             stats[17] = _summarize(0, iouThr=.5, areaRng='large',
#                                    maxDets=self.params.maxDets[2])
#             return stats
#
#         def _summarizeKps():
#             stats = np.zeros((10,))
#             stats[0] = _summarize(1, maxDets=20)
#             stats[1] = _summarize(1, maxDets=20, iouThr=.5)
#             stats[2] = _summarize(1, maxDets=20, iouThr=.75)
#             stats[3] = _summarize(1, maxDets=20, areaRng='medium')
#             stats[4] = _summarize(1, maxDets=20, areaRng='large')
#             stats[5] = _summarize(0, maxDets=20)
#             stats[6] = _summarize(0, maxDets=20, iouThr=.5)
#             stats[7] = _summarize(0, maxDets=20, iouThr=.75)
#             stats[8] = _summarize(0, maxDets=20, areaRng='medium')
#             stats[9] = _summarize(0, maxDets=20, areaRng='large')
#             return stats
#
#         if not self.eval:
#             raise Exception('Please run accumulate() first')
#         iouType = self.params.iouType
#         if iouType == 'segm' or iouType == 'bbox':
#             summarize = _summarizeDets
#         elif iouType == 'keypoints':
#             summarize = _summarizeKps
#         self.stats = summarize()
#
#     print("HACKING: overriding COCOeval.summarize = boulder_summarize...")
#     COCOeval.summarize = boulder_summarize
#
#     class BoulderEvaluator(DatasetEvaluator):
#         """
#         Evaluate AR for object proposals, AP for instance detection/segmentation, AP
#         for keypoint detection outputs using COCO's metrics.
#         See http://cocodataset.org/#detection-eval and
#         http://cocodataset.org/#keypoints-eval to understand its metrics.
#
#         In addition to COCO, this evaluator is able to support any bounding box detection,
#         instance segmentation, or keypoint detection dataset.
#         """
#
#         def __init__(
#                 self,
#                 dataset_name,
#                 tasks=None,
#                 distributed=True,
#                 output_dir=None,
#                 *,
#                 max_dets_per_image=None,
#                 use_fast_impl=True,
#                 kpt_oks_sigmas=(),
#         ):
#             """
#             Args:
#                 dataset_name (str): name of the dataset to be evaluated.
#                     It must have either the following corresponding metadata:
#
#                         "json_file": the path to the COCO format annotation
#
#                     Or it must be in detectron2's standard dataset format
#                     so it can be converted to COCO format automatically.
#                 tasks (tuple[str]): tasks that can be evaluated under the given
#                     configuration. A task is one of "bbox", "segm", "keypoints".
#                     By default, will infer this automatically from predictions.
#                 distributed (True): if True, will collect results from all ranks and run evaluation
#                     in the main process.
#                     Otherwise, will only evaluate the results in the current process.
#                 output_dir (str): optional, an output directory to dump all
#                     results predicted on the dataset. The dump contains two files:
#
#                     1. "instances_predictions.pth" a file in torch serialization
#                        format that contains all the raw original predictions.
#                     2. "coco_instances_results.json" a json file in COCO's result
#                        format.
#                 use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
#                     Although the results should be very close to the official implementation in COCO
#                     API, it is still recommended to compute results with the official API for use in
#                     papers. The faster implementation also uses more RAM.
#                 kpt_oks_sigmas (list[float]): The sigmas used to calculate keypoint OKS.
#                     See http://cocodataset.org/#keypoints-eval
#                     When empty, it will use the defaults in COCO.
#                     Otherwise it should be the same length as ROI_KEYPOINT_HEAD.NUM_KEYPOINTS.
#             """
#             self._logger = logging.getLogger(__name__)
#             self._distributed = distributed
#             self._output_dir = output_dir
#             self._use_fast_impl = use_fast_impl
#
#             if max_dets_per_image is None:
#                 max_dets_per_image = [1, 10, 100]
#             else:
#                 max_dets_per_image = [1, 10, max_dets_per_image]
#             self._max_dets_per_image = max_dets_per_image
#
#             if tasks is not None and isinstance(tasks, CfgNode):
#                 kpt_oks_sigmas = (
#                     tasks.TEST.KEYPOINT_OKS_SIGMAS if not kpt_oks_sigmas else kpt_oks_sigmas
#                 )
#                 self._logger.warn(
#                     "COCO Evaluator instantiated using config, this is deprecated behavior."
#                     " Please pass in explicit arguments instead."
#                 )
#                 self._tasks = None  # Infering it from predictions should be better
#             else:
#                 self._tasks = tasks
#
#             self._cpu_device = torch.device("cpu")
#
#             self._metadata = MetadataCatalog.get(dataset_name)
#             if not hasattr(self._metadata, "json_file"):
#                 self._logger.info(
#                     f"'{dataset_name}' is not registered by `register_coco_instances`."
#                     " Therefore trying to convert it to COCO format ..."
#                 )
#
#                 cache_path = os.path.join(output_dir,
#                                           f"{dataset_name}_coco_format.json")
#                 self._metadata.json_file = cache_path
#                 convert_to_coco_json(dataset_name, cache_path)
#
#             json_file = PathManager.get_local_path(self._metadata.json_file)
#             with contextlib.redirect_stdout(io.StringIO()):
#                 self._coco_api = COCO(json_file)
#
#             # Test set json files do not contain annotations (evaluation must be
#             # performed using the COCO evaluation server).
#             self._do_evaluation = "annotations" in self._coco_api.dataset
#             if self._do_evaluation:
#                 self._kpt_oks_sigmas = kpt_oks_sigmas
#
#         def reset(self):
#             self._predictions = []
#
#         def process(self, inputs, outputs):
#             """
#             Args:
#                 inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
#                     It is a list of dict. Each dict corresponds to an image and
#                     contains keys like "height", "width", "file_name", "image_id".
#                 outputs: the outputs of a COCO model. It is a list of dicts with key
#                     "instances" that contains :class:`Instances`.
#             """
#             for input, output in zip(inputs, outputs):
#                 prediction = {"image_id": input["image_id"]}
#
#                 if "instances" in output:
#                     instances = output["instances"].to(self._cpu_device)
#                     prediction["instances"] = instances_to_coco_json(instances,
#                                                                      input[
#                                                                          "image_id"])
#                 if "proposals" in output:
#                     prediction["proposals"] = output["proposals"].to(
#                         self._cpu_device)
#                 if len(prediction) > 1:
#                     self._predictions.append(prediction)
#
#         def evaluate(self, img_ids=None):
#             """
#             Args:
#                 img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
#             """
#             if self._distributed:
#                 comm.synchronize()
#                 predictions = comm.gather(self._predictions, dst=0)
#                 predictions = list(itertools.chain(*predictions))
#
#                 if not comm.is_main_process():
#                     return {}
#             else:
#                 predictions = self._predictions
#
#             if len(predictions) == 0:
#                 self._logger.warning(
#                     "[VinbigdataEvaluator] Did not receive valid predictions.")
#                 return {}
#
#             if self._output_dir:
#                 PathManager.mkdirs(self._output_dir)
#                 file_path = os.path.join(self._output_dir,
#                                          "instances_predictions.pth")
#                 with PathManager.open(file_path, "wb") as f:
#                     torch.save(predictions, f)
#
#             self._results = OrderedDict()
#             if "proposals" in predictions[0]:
#                 self._eval_box_proposals(predictions)
#             if "instances" in predictions[0]:
#                 self._eval_predictions(predictions, img_ids=img_ids)
#             # Copy so the caller can do whatever with results
#             return copy.deepcopy(self._results)
#
#         def _tasks_from_predictions(self, predictions):
#             """
#             Get COCO API "tasks" (i.e. iou_type) from COCO-format predictions.
#             """
#             tasks = {"bbox"}
#             for pred in predictions:
#                 if "segmentation" in pred:
#                     tasks.add("segm")
#                 if "keypoints" in pred:
#                     tasks.add("keypoints")
#             return sorted(tasks)
#
#         def _eval_predictions(self, predictions, img_ids=None):
#             """
#             Evaluate predictions. Fill self._results with the metrics of the tasks.
#             """
#             self._logger.info("Preparing results for COCO format ...")
#             coco_results = list(
#                 itertools.chain(*[x["instances"] for x in predictions]))
#             tasks = self._tasks or self._tasks_from_predictions(coco_results)
#
#             # unmap the category ids for COCO
#             if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
#                 dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
#                 all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
#                 num_classes = len(all_contiguous_ids)
#                 assert min(all_contiguous_ids) == 0 and max(
#                     all_contiguous_ids) == num_classes - 1
#
#                 reverse_id_mapping = {v: k for k, v in
#                                       dataset_id_to_contiguous_id.items()}
#                 for result in coco_results:
#                     category_id = result["category_id"]
#                     assert category_id < num_classes, (
#                         f"A prediction has class={category_id}, "
#                         f"but the dataset only has {num_classes} classes and "
#                         f"predicted class id should be in [0, {num_classes - 1}]."
#                     )
#                     result["category_id"] = reverse_id_mapping[category_id]
#
#             if self._output_dir:
#                 file_path = os.path.join(self._output_dir,
#                                          "coco_instances_results.json")
#                 self._logger.info("Saving results to {}".format(file_path))
#                 with PathManager.open(file_path, "w") as f:
#                     f.write(json.dumps(coco_results))
#                     f.flush()
#
#             if not self._do_evaluation:
#                 self._logger.info(
#                     "Annotations are not available for evaluation.")
#                 return
#
#             self._logger.info(
#                 "Evaluating predictions with {} COCO API...".format(
#                     "unofficial" if self._use_fast_impl else "official"
#                 )
#             )
#             for task in sorted(tasks):
#                 coco_eval = (
#                     _evaluate_predictions_on_coco(
#                         self._coco_api,
#                         coco_results,
#                         task,
#                         kpt_oks_sigmas=self._kpt_oks_sigmas,
#                         use_fast_impl=self._use_fast_impl,
#                         img_ids=img_ids,
#                         max_dets_per_image=self._max_dets_per_image,
#                     )
#                     if len(coco_results) > 0
#                     else None  # cocoapi does not handle empty results very well
#                 )
#
#                 res = self._derive_coco_results(
#                     coco_eval, task,
#                     class_names=self._metadata.get("thing_classes")
#                 )
#                 self._results[task] = res
#
#         def _eval_box_proposals(self, predictions):
#             """
#             Evaluate the box proposals in predictions.
#             Fill self._results with the metrics for "box_proposals" task.
#             """
#             if self._output_dir:
#                 # Saving generated box proposals to file.
#                 # Predicted box_proposals are in XYXY_ABS mode.
#                 bbox_mode = BoxMode.XYXY_ABS.value
#                 ids, boxes, objectness_logits = [], [], []
#                 for prediction in predictions:
#                     ids.append(prediction["image_id"])
#                     boxes.append(
#                         prediction["proposals"].proposal_boxes.tensor.numpy())
#                     objectness_logits.append(
#                         prediction["proposals"].objectness_logits.numpy())
#
#                 proposal_data = {
#                     "boxes": boxes,
#                     "objectness_logits": objectness_logits,
#                     "ids": ids,
#                     "bbox_mode": bbox_mode,
#                 }
#                 with PathManager.open(
#                         os.path.join(self._output_dir, "box_proposals.pkl"),
#                         "wb") as f:
#                     pickle.dump(proposal_data, f)
#
#             if not self._do_evaluation:
#                 self._logger.info(
#                     "Annotations are not available for evaluation.")
#                 return
#
#             self._logger.info("Evaluating bbox proposals ...")
#             res = {}
#             areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
#             for limit in [100, 1000]:
#                 for area, suffix in areas.items():
#                     stats = _evaluate_box_proposals(predictions, self._coco_api,
#                                                     area=area, limit=limit)
#                     key = "AR{}@{:d}".format(suffix, limit)
#                     res[key] = float(stats["ar"].item() * 100)
#             self._logger.info("Proposal metrics: \n" + create_small_table(res))
#             self._results["box_proposals"] = res
#
#         def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
#             """
#             Derive the desired score numbers from summarized COCOeval.
#
#             Args:
#                 coco_eval (None or COCOEval): None represents no predictions from model.
#                 iou_type (str):
#                 class_names (None or list[str]): if provided, will use it to predict
#                     per-category AP.
#
#             Returns:
#                 a dict of {metric name: score}
#             """
#
#             metrics = {
#                 "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
#                 "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
#                 "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
#             }[iou_type]
#
#             if coco_eval is None:
#                 self._logger.warn("No predictions from the model!")
#                 return {metric: float("nan") for metric in metrics}
#
#             # the standard metrics
#             results = {
#                 metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[
#                                                                 idx] >= 0 else "nan")
#                 for idx, metric in enumerate(metrics)
#             }
#             self._logger.info(
#                 "Evaluation results for {}: \n".format(
#                     iou_type) + create_small_table(results)
#             )
#             if not np.isfinite(sum(results.values())):
#                 self._logger.info(
#                     "Some metrics cannot be computed and is shown as NaN.")
#
#             if class_names is None or len(class_names) <= 1:
#                 return results
#             # Compute per-category AP
#             # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
#             precisions = coco_eval.eval["precision"]
#             # precision has dims (iou, recall, cls, area range, max dets)
#             assert len(class_names) == precisions.shape[2]
#
#             results_per_category = []
#             for idx, name in enumerate(class_names):
#                 # area range index 0: all area ranges
#                 # max dets index -1: typically 100 per image
#                 precision = precisions[:, :, idx, 0, -1]
#                 precision = precision[precision > -1]
#                 ap = np.mean(precision) if precision.size else float("nan")
#                 results_per_category.append(
#                     ("{}".format(name), float(ap * 100)))
#
#             # tabulate it
#             N_COLS = min(6, len(results_per_category) * 2)
#             results_flatten = list(itertools.chain(*results_per_category))
#             results_2d = itertools.zip_longest(
#                 *[results_flatten[i::N_COLS] for i in range(N_COLS)])
#             table = tabulate(
#                 results_2d,
#                 tablefmt="pipe",
#                 floatfmt=".3f",
#                 headers=["category", "AP"] * (N_COLS // 2),
#                 numalign="left",
#             )
#             self._logger.info("Per-category {} AP: \n".format(iou_type) + table)
#
#             results.update(
#                 {"AP-" + name: ap for name, ap in results_per_category})
#             return results
#
#     def instances_to_coco_json(instances, img_id):
#         """
#         Dump an "Instances" object to a COCO-format json that's used for evaluation.
#
#         Args:
#             instances (Instances):
#             img_id (int): the image id
#
#         Returns:
#             list[dict]: list of json annotations in COCO format.
#         """
#         num_instance = len(instances)
#         if num_instance == 0:
#             return []
#
#         boxes = instances.pred_boxes.tensor.numpy()
#         boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
#         boxes = boxes.tolist()
#         scores = instances.scores.tolist()
#         classes = instances.pred_classes.tolist()
#
#         has_mask = instances.has("pred_masks")
#         if has_mask:
#             # use RLE to encode the masks, because they are too large and takes memory
#             # since this evaluator stores outputs of the entire dataset
#             rles = [
#                 mask_util.encode(
#                     np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
#                 for mask in instances.pred_masks
#             ]
#             for rle in rles:
#                 # "counts" is an array encoded by mask_util as a byte-stream. Python3's
#                 # json writer which always produces strings cannot serialize a bytestream
#                 # unless you decode it. Thankfully, utf-8 works out (which is also what
#                 # the pycocotools/_mask.pyx does).
#                 rle["counts"] = rle["counts"].decode("utf-8")
#
#         has_keypoints = instances.has("pred_keypoints")
#         if has_keypoints:
#             keypoints = instances.pred_keypoints
#
#         results = []
#         for k in range(num_instance):
#             result = {
#                 "image_id": img_id,
#                 "category_id": classes[k],
#                 "bbox": boxes[k],
#                 "score": scores[k],
#             }
#             if has_mask:
#                 result["segmentation"] = rles[k]
#             if has_keypoints:
#                 # In COCO annotations,
#                 # keypoints coordinates are pixel indices.
#                 # However our predictions are floating point coordinates.
#                 # Therefore we subtract 0.5 to be consistent with the annotation format.
#                 # This is the inverse of data loading logic in `datasets/coco.py`.
#                 keypoints[k][:, :2] -= 0.5
#                 result["keypoints"] = keypoints[k].flatten().tolist()
#             results.append(result)
#         return results
#
#     # inspired from Detectron:
#     # https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L255 # noqa
#     def _evaluate_box_proposals(dataset_predictions, coco_api, thresholds=None,
#                                 area="all", limit=None):
#         """
#         Evaluate detection proposal recall metrics. This function is a much
#         faster alternative to the official COCO API recall evaluation code. However,
#         it produces slightly different results.
#         """
#         # Record max overlap value for each gt box
#         # Return vector of overlap values
#         areas = {
#             "all": 0,
#             "small": 1,
#             "medium": 2,
#             "large": 3,
#             "96-128": 4,
#             "128-256": 5,
#             "256-512": 6,
#             "512-inf": 7,
#         }
#         area_ranges = [
#             [6 ** 2, 1e5 ** 2],  # all
#             [6 ** 2, 16 ** 2],  # small
#             # [16 ** 2, 32 ** 2],  # small
#             [16 ** 2, 32 ** 2],  # medium
#             [32 ** 2, 1e5 ** 2],  # large
#             [96 ** 2, 128 ** 2],  # 96-128
#             [128 ** 2, 256 ** 2],  # 128-256
#             [256 ** 2, 512 ** 2],  # 256-512
#             [512 ** 2, 1e5 ** 2],
#         ]  # 512-inf
#         assert area in areas, "Unknown area range: {}".format(area)
#         area_range = area_ranges[areas[area]]
#         gt_overlaps = []
#         num_pos = 0
#
#         for prediction_dict in dataset_predictions:
#             predictions = prediction_dict["proposals"]
#
#             # sort predictions in descending order
#             # TODO maybe remove this and make it explicit in the documentation
#             inds = predictions.objectness_logits.sort(descending=True)[1]
#             predictions = predictions[inds]
#
#             ann_ids = coco_api.getAnnIds(imgIds=prediction_dict["image_id"])
#             anno = coco_api.loadAnns(ann_ids)
#             gt_boxes = [
#                 BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
#                 for obj in anno
#                 if obj["iscrowd"] == 0
#             ]
#             gt_boxes = torch.as_tensor(gt_boxes).reshape(-1,
#                                                          4)  # guard against no boxes
#             gt_boxes = Boxes(gt_boxes)
#             gt_areas = torch.as_tensor(
#                 [obj["area"] for obj in anno if obj["iscrowd"] == 0])
#
#             if len(gt_boxes) == 0 or len(predictions) == 0:
#                 continue
#
#             valid_gt_inds = (gt_areas >= area_range[0]) & (
#                     gt_areas <= area_range[1])
#             gt_boxes = gt_boxes[valid_gt_inds]
#
#             num_pos += len(gt_boxes)
#
#             if len(gt_boxes) == 0:
#                 continue
#
#             if limit is not None and len(predictions) > limit:
#                 predictions = predictions[:limit]
#
#             overlaps = pairwise_iou(predictions.proposal_boxes, gt_boxes)
#
#             _gt_overlaps = torch.zeros(len(gt_boxes))
#             for j in range(min(len(predictions), len(gt_boxes))):
#                 # find which proposal box maximally covers each gt box
#                 # and get the iou amount of coverage for each gt box
#                 max_overlaps, argmax_overlaps = overlaps.max(dim=0)
#
#                 # find which gt box is 'best' covered (i.e. 'best' = most iou)
#                 gt_ovr, gt_ind = max_overlaps.max(dim=0)
#                 assert gt_ovr >= 0
#                 # find the proposal box that covers the best covered gt box
#                 box_ind = argmax_overlaps[gt_ind]
#                 # record the iou coverage of this gt box
#                 _gt_overlaps[j] = overlaps[box_ind, gt_ind]
#                 assert _gt_overlaps[j] == gt_ovr
#                 # mark the proposal box and the gt box as used
#                 overlaps[box_ind, :] = -1
#                 overlaps[:, gt_ind] = -1
#
#             # append recorded iou coverage level
#             gt_overlaps.append(_gt_overlaps)
#         gt_overlaps = (
#             torch.cat(gt_overlaps, dim=0) if len(gt_overlaps) else torch.zeros(
#                 0,
#                 dtype=torch.float32)
#         )
#         gt_overlaps, _ = torch.sort(gt_overlaps)
#
#         if thresholds is None:
#             step = 0.05
#             thresholds = torch.arange(0.5, 0.95 + 1e-5, step,
#                                       dtype=torch.float32)
#             # thresholds = torch.arange(0.4, 0.95 + 1e-5, step, dtype=torch.float32)
#         recalls = torch.zeros_like(thresholds)
#         # compute recall for each iou threshold
#         for i, t in enumerate(thresholds):
#             recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
#         # ar = 2 * np.trapz(recalls, thresholds)
#         ar = recalls.mean()
#         return {
#             "ar": ar,
#             "recalls": recalls,
#             "thresholds": thresholds,
#             "gt_overlaps": gt_overlaps,
#             "num_pos": num_pos,
#         }
#
#     def _evaluate_predictions_on_coco(
#             coco_gt, coco_results, iou_type, kpt_oks_sigmas=None,
#             use_fast_impl=True, img_ids=None, max_dets_per_image=None
#     ):
#         """
#         Evaluate the coco results using COCOEval API.
#         """
#         assert len(coco_results) > 0
#
#         if iou_type == "segm":
#             coco_results = copy.deepcopy(coco_results)
#             # When evaluating mask AP, if the results contain bbox, cocoapi will
#             # use the box area as the area of the instance, instead of the mask area.
#             # This leads to a different definition of small/medium/large.
#             # We remove the bbox field to let mask AP use mask area.
#             for c in coco_results:
#                 c.pop("bbox", None)
#
#         coco_dt = coco_gt.loadRes(coco_results)
#         coco_eval = (COCOeval_opt if use_fast_impl else COCOeval)(coco_gt,
#                                                                   coco_dt,
#                                                                   iou_type)
#
#         # HACKING: overwrite areaRng
#         coco_eval.params.areaRng = [[6 ** 2, 1e5 ** 2], [6 ** 2, 16 ** 2],
#                                     [16 ** 2, 32 ** 2], [32 ** 2, 1e5 ** 2]]
#         # coco_eval.params.areaRng = [[6 ** 2, 1e5 ** 2], [16 ** 2, 32 ** 2], [16 ** 2, 32 ** 2], [32 ** 2, 1e5 ** 2]]
#
#         # For COCO, the default max_dets_per_image is [1, 10, 100].
#         if max_dets_per_image is None:
#             max_dets_per_image = [1, 10, 100]  # Default from COCOEval
#         else:
#             assert (
#                     len(max_dets_per_image) >= 3
#             ), "COCOeval requires maxDets (and max_dets_per_image) to have length at least 3"
#             # In the case that user supplies a custom input for max_dets_per_image,
#             # apply COCOevalMaxDets to evaluate AP with the custom input.
#             if max_dets_per_image[2] != 100:
#                 None
#                 # coco_eval = COCOevalMaxDets(coco_gt, coco_dt,
#                 #                            iou_type)  # this need to be changed
#
#         if iou_type != "keypoints":
#             coco_eval.params.maxDets = max_dets_per_image
#
#         if img_ids is not None:
#             coco_eval.params.imgIds = img_ids
#
#         if iou_type == "keypoints":
#             # Use the COCO default keypoint OKS sigmas unless overrides are specified
#             if kpt_oks_sigmas:
#                 assert hasattr(coco_eval.params,
#                                "kpt_oks_sigmas"), "pycocotools is too old!"
#                 coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)
#             # COCOAPI requires every detection and every gt to have keypoints, so
#             # we just take the first entry from both
#             num_keypoints_dt = len(coco_results[0]["keypoints"]) // 3
#             num_keypoints_gt = len(
#                 next(iter(coco_gt.anns.values()))["keypoints"]) // 3
#             num_keypoints_oks = len(coco_eval.params.kpt_oks_sigmas)
#             assert num_keypoints_oks == num_keypoints_dt == num_keypoints_gt, (
#                 f"[BoulderEvaluator] Prediction contain {num_keypoints_dt} keypoints. "
#                 f"Ground truth contains {num_keypoints_gt} keypoints. "
#                 f"The length of cfg.TEST.KEYPOINT_OKS_SIGMAS is {num_keypoints_oks}. "
#                 "They have to agree with each other. For meaning of OKS, please refer to "
#                 "http://cocodataset.org/#keypoints-eval."
#             )
#
#         coco_eval.evaluate()
#         coco_eval.accumulate()
#         coco_eval.summarize()
#
#         return coco_eval
#
#     class MyTrainer(DefaultTrainer):
#         @classmethod
#         def build_train_loader(cls, cfg, is_train=True, sampler=None):
#             return build_detection_train_loader(
#                 cfg, mapper=DatasetMapper(cfg, is_train), sampler=sampler
#             )
#
#         @classmethod
#         def build_test_loader(cls, cfg, dataset_name):
#             return build_detection_test_loader(
#                 cfg, dataset_name, mapper=DatasetMapper(cfg, False)
#             )
#
#         @classmethod
#         def build_evaluator(cls, cfg, dataset_name):
#             output_folder = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
#             os.makedirs(output_folder, exist_ok=True)
#             return BoulderEvaluator(dataset_name, ("segm",), False,
#                                  output_folder, max_dets_per_image=1000)
#
#     @dataclass
#     class BOULDERconfig:
#         # augmentations
#         aug_kwargs: Dict = field(default_factory=lambda: {})
#
#         def update(self, param_dict: Dict) -> "BOULDERconfig":
#             # Overwrite by `param_dict`
#             for key, value in param_dict.items():
#                 if not hasattr(self, key):
#                     raise ValueError(
#                         f"[ERROR] Unexpected key for flag = {key}")
#                 setattr(self, key, value)
#             return self
#
#     class ValidationLoss(HookBase):
#         def __init__(self, cfg):
#             super().__init__()
#             self.cfg = cfg.clone()
#             self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST[0]
#             self._loader = iter(
#                 MyTrainer.build_train_loader(self.cfg, is_train=False))  # False, for not applying any transforms
#
#         def after_step(self):
#             data = next(self._loader)
#             with torch.no_grad():
#                 loss_dict = self.trainer.model(data)
#
#                 losses = sum(loss_dict.values())
#                 assert torch.isfinite(losses).all(), loss_dict
#
#                 loss_dict_reduced = {"val_" + k: v.item() for k, v in
#                                      comm.reduce_dict(loss_dict).items()}
#                 losses_reduced = sum(
#                     loss for loss in loss_dict_reduced.values())
#                 if comm.is_main_process():
#                     self.trainer.storage.put_scalars(
#                         total_val_loss=losses_reduced,
#                         **loss_dict_reduced)
#
#     # read augmentations
#     augmentations_dict = load_yaml(augmentation_file)
#     flags = BOULDERconfig().update(augmentations_dict)
#     cfg = get_cfg()
#     cfg.merge_from_file(config_file)
#
#     # Save complete config file
#     with open(config_file_complete, "w") as f:
#       f.write(cfg.dump())
#
#     cfg.aug_kwargs = CN(flags.aug_kwargs)  # pass aug_kwargs to cfg
#     cfg.MODEL.DEVICE = device
#
#     # training
#     trainer = MyTrainer(cfg)
#     val_loss = ValidationLoss(cfg)
#     trainer.register_hooks([val_loss])
#     # swap the order of PeriodicWriter and ValidationLoss
#     trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
#     trainer.resume_or_load(resume=False)
#     trainer.train()