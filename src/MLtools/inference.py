import albumentations as A
import geopandas as gpd
import numpy as np
import rasterio as rio

import pandas as pd
import torch
import torchvision
import sys

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import detection_utils as utils

from pathlib import Path
from rasterio import features
from rastertools_BOULDERING import metadata as raster_metadata, raster
from tqdm import tqdm

#TODO: move spatial_selection() and searching_area() to shptools
def predict(config_file, model_weights, device, image_dir, out_shapefile,
            search_pattern, scores_thresh_test, nms_thresh_test,
            min_size_test, max_size_test,
            pre_nms_topk_test, post_nms_topk_test,
            detections_per_image):
    """
    Description of params for scores, min_size_test, max_size_test,
    pre_nms_topk_test, post_nms_topk_test and detections_per_image are copied
    from the Detectron2 config file explanation (see
    https://detectron2.readthedocs.io/en/latest/modules/config.html).

    For the detection boulders, the parameters give pretty good results:
    scores_thresh_test = 0.10 (before 0.50)
    nms_thresh_test = 0.30 (0.50)
    min_size_test = 1024 (800)
    max_size_test = 1024 (800)
    pre_nms_topk_test = 2000 (?)
    post_nms_topk_test = 1000 (or 500)
    detections_per_image = 2000 (in case you have lot of boulders in your image)

    :param config_file:
    :param model_weights:
    :param device:
    :param image_dir:
    :param out_shapefile:
    :param search_pattern:
    :param scores_thresh_test: Minimum score threshold (assuming scores in a
    [0, 1] range); a value chosen to balance obtaining high recall with not
    having too many low precision detections that will slow down inference
    post processing steps (like NMS). A default threshold of 0.0 increases AP
    by ~0.2-0.3 but significantly slows down inference.
    :param nms_thresh_test: Overlap threshold used for non-maximum suppression
    (suppress boxes with IoU >= this threshold).
    :param min_size_test: Size of the smallest side of the image during testing.
    Set to zero to disable resize in testing.
    :param max_size_test: Maximum size of the side of the image during testing
    :param pre_nms_topk_test: Number of top scoring RPN proposals to keep before
    applying NMS When FPN is used, this is *per FPN level* (not total)
    :param post_nms_topk_test: Number of top scoring RPN proposals to keep after
    applying NMS When FPN is used, this limit is applied per level and then again
    to the union of proposals from all levels. NOTE: When FPN is used, the
    meaning of this config is different from Detectron1. It means per-batch topk
    in Detectron1, but per-image topk here. See the "find_top_rpn_proposals"
    function for details.
    :param detections_per_image: Maximum number of detections to return per
    image during inference

    connectivity = 4 by default in features.shape?

    :return:
    """
    # load model and weight of the models
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_weights.as_posix()
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = scores_thresh_test
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_thresh_test
    cfg.INPUT.MIN_SIZE_TEST = min_size_test
    cfg.INPUT.MAX_SIZE_TEST = max_size_test
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = pre_nms_topk_test
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = post_nms_topk_test
    cfg.TEST.DETECTIONS_PER_IMAGE = detections_per_image


    # predictor for one image
    predictor = DefaultPredictor(cfg)

    image_dir = Path(image_dir)
    geoms = []
    scores_list = []
    boulder_id = []

    bid = 0

    for tif in tqdm(sorted(image_dir.glob(search_pattern))):
        #print(tif.name)
        png = tif.with_name(tif.name.replace('tif', 'png'))

        with rio.open(tif) as src:
            meta = src.meta

            # loading image
            array = utils.read_image(png, format="BGR")

            # I could actually double the size of the image here

            # inference
            outputs = predictor(array)

            for i, pred in enumerate(outputs["instances"].pred_masks.to("cpu")):
                pred_mask = torch.Tensor.numpy(pred)
                pred_mask = (pred_mask + 0.0).astype('uint8')
                results = (
                    {'properties': {'raster_val': v}, 'geometry': s}
                    for j, (s, v)
                    in enumerate(
                    features.shapes(pred_mask, mask=pred_mask,
                                    transform=src.transform)))

                results_ = list(results)
                # no predictions
                if len(results_) == 0:
                    None
                else:
                    # this is necessary as sometimes multipolygons are generated
                    for res in results_:
                        geoms.append(res)
                        boulder_id.append(bid)
                        bid = bid + 1
                        scores_list.append(float(torch.Tensor.numpy(outputs["instances"].scores.to("cpu")[i])))

    if len(geoms) > 0:
        gpd_polygonized_raster = gpd.GeoDataFrame.from_features(geoms, crs=meta["crs"])
        gpd_polygonized_raster["scores"] = scores_list
        gpd_polygonized_raster["boulder_id"] = boulder_id
        gpd_polygonized_raster.to_file(out_shapefile)

    # no predictions in the whole image!
    else:
        schema = {"geometry": "Polygon",
                  "properties": {"raster_val": "float", "scores": "float",
                                 "boulder_id": "int"}}
        gdf_empty = gpd.GeoDataFrame(geometry=[])
        gdf_empty.to_file(out_shapefile, driver='ESRI Shapefile', schema=schema, crs=meta["crs"])


def predict_tta(config_file, model_weights, device, image_dir, out_shapefile,
                search_pattern, scores_thresh_test, nms_thresh_test,
                min_size_test, max_size_test,
                pre_nms_topk_test, post_nms_topk_test,
                detections_per_image):
    """
    Description of params for scores, min_size_test, max_size_test,
    pre_nms_topk_test, post_nms_topk_test and detections_per_image are copied
    from the Detectron2 config file explanation (see
    https://detectron2.readthedocs.io/en/latest/modules/config.html).

    For the detection boulders, the parameters give pretty good results:
    scores_thresh_test = 0.10 (before 0.50)
    nms_thresh_test = 0.30 (0.50)
    min_size_test = 1024 (800)
    max_size_test = 1024 (800)
    pre_nms_topk_test = 2000 (?)
    post_nms_topk_test = 1000 (or 500)
    detections_per_image = 2000 (in case you have lot of boulders in your image)

    :param config_file:
    :param model_weights:
    :param device:
    :param image_dir:
    :param out_shapefile:
    :param search_pattern:
    :param scores_thresh_test: Minimum score threshold (assuming scores in a
    [0, 1] range); a value chosen to balance obtaining high recall with not
    having too many low precision detections that will slow down inference
    post processing steps (like NMS). A default threshold of 0.0 increases AP
    by ~0.2-0.3 but significantly slows down inference.
    :param nms_thresh_test: Overlap threshold used for non-maximum suppression
    (suppress boxes with IoU >= this threshold).
    :param min_size_test: Size of the smallest side of the image during testing.
    Set to zero to disable resize in testing.
    :param max_size_test: Maximum size of the side of the image during testing
    :param pre_nms_topk_test: Number of top scoring RPN proposals to keep before
    applying NMS When FPN is used, this is *per FPN level* (not total)
    :param post_nms_topk_test: Number of top scoring RPN proposals to keep after
    applying NMS When FPN is used, this limit is applied per level and then again
    to the union of proposals from all levels. NOTE: When FPN is used, the
    meaning of this config is different from Detectron1. It means per-batch topk
    in Detectron1, but per-image topk here. See the "find_top_rpn_proposals"
    function for details.
    :param detections_per_image: Maximum number of detections to return per
    image during inference

    is connectivity = 4 in features.shape?

    :return:
    """

    # load model and weight of the models
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_weights.as_posix()
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = scores_thresh_test
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_thresh_test
    cfg.INPUT.MIN_SIZE_TEST = min_size_test
    cfg.INPUT.MAX_SIZE_TEST = max_size_test
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = pre_nms_topk_test
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = post_nms_topk_test
    cfg.TEST.DETECTIONS_PER_IMAGE = detections_per_image

    # predictor for one image
    predictor = DefaultPredictor(cfg)
    image_dir = Path(image_dir)
    gdfs_list = []

    # augmentations
    transforms = [A.NoOp(p=1.0), A.Affine(p=1.0, rotate=90.0),
                  A.Affine(p=1.0, rotate=180.0), A.Affine(p=1.0, rotate=270.0),
                  A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0),
                  A.Transpose(p=1.0), A.Compose(
            [A.Affine(p=1.0, rotate=180.0), A.Transpose(p=1.0)])]

    inverse_transforms = [A.NoOp(p=1.0), A.Affine(p=1.0, rotate=-90.0),
                          A.Affine(p=1.0, rotate=-180.0),
                          A.Affine(p=1.0, rotate=-270.0),
                          A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0),
                          A.Transpose(p=1.0), A.Compose(
            [A.Affine(p=1.0, rotate=180.0), A.Transpose(p=1.0)])]

    for tif in tqdm(sorted(image_dir.glob(search_pattern))):
        png = tif.with_name(tif.name.replace('tif', 'png'))
        geoms = []
        scores_list = []

        with rio.open(tif) as src:
            meta = src.meta

            # loading image
            array = utils.read_image(png, format="BGR")

            for k, t in enumerate(transforms):

                # inference
                transformed = t(image=array)
                image_tranformed = transformed["image"]
                outputs = predictor(image_tranformed)
                masks = torch.Tensor.numpy(
                    outputs["instances"].pred_masks.to("cpu")).astype('uint8')

                # inverse transforms
                t_inv = inverse_transforms[k]
                inv_transformed = t_inv(image=image_tranformed,
                                        masks=list(masks))
                mask_inverse_transformed = np.array(inv_transformed["masks"])

                for i, pred_mask in enumerate(mask_inverse_transformed):
                    results = (
                        {'properties': {'raster_val': v}, 'geometry': s}
                        for j, (s, v)
                        in enumerate(
                        features.shapes(pred_mask, mask=pred_mask,
                                        transform=src.transform)))

                    results_ = list(results)
                    # no predictions
                    if len(results_) == 0:
                        None
                    else:
                        # this is necessary as sometimes multipolygons are generated (maybe I should get rid of those situations!)
                        for res in results_:
                            geoms.append(res)
                            scores_list.append(float(torch.Tensor.numpy(
                                outputs["instances"].scores.to("cpu")[i])))

            if len(geoms) > 0:
                gdf_polygonized_raster = gpd.GeoDataFrame.from_features(geoms, crs=meta["crs"])
                gdf_polygonized_raster["scores"] = scores_list
                gdf_nms = nms(gdf_polygonized_raster, nms_thresh_test)
                gdfs_list.append(gdf_nms)
            # no predictions in the whole image!
            else:
                None

    if len(gdfs_list) > 0:
        gdf_total = gpd.GeoDataFrame(pd.concat(gdfs_list, ignore_index=True))
        gdf_total["boulder_id"] = np.arange(gdf_total.shape[0]).astype("int")
        gdf_total.to_file(out_shapefile)
    else:

        schema = {"geometry": "Polygon",
                  "properties": {"raster_val": "float", "scores": "float",
                                 "boulder_id": "int"}}
        gdf_total = gpd.GeoDataFrame(geometry=[])
        gdf_total.to_file(out_shapefile, driver='ESRI Shapefile', schema=schema, crs=meta["crs"])


def predictions(in_raster, config_file, model_weights, device,
                search_tif_pattern,
                block_width, block_height, output_dir, is_tta=True, scores_thresh_test=0.10,
                nms_thresh_test=0.30,
                min_size_test=512, max_size_test=512, pre_nms_topk_test=2000,
                post_nms_topk_test=1000,
                detections_per_image=2000):
    """
    Remember that (height, width)
    """

    in_raster = Path(in_raster)
    output_dir = Path(output_dir)
    scores_str = "scores" + str(int(scores_thresh_test * 100)).zfill(3)
    config_version = \
    [i for i in config_file.stem.split("-") if i.startswith("v0")][0]  # name dependent which is not good...
    min_size_test_str = "minsize" + str(int(min_size_test)).zfill(4)
    max_size_test_str = "maxsize" + str(int(max_size_test)).zfill(4)

    assert block_width % 2 == 0, "Please chose a block_width that can be divived by two."
    assert block_height % 2 == 0, "Please chose a block_height that can be divived by two."
    assert block_width == block_height, "Current version of the edge-fixing algorithm works only for square image patches."

    # create grids
    stride_widths = [0, int(block_width / 2), int(block_width / 2), block_width,
                     0, int(block_width / 2)]
    stride_heights = [0, int(block_height / 2), block_height,
                      int(block_height / 2), int(block_height / 2), 0]

    # create path
    (output_dir / "shp").mkdir(parents=True, exist_ok=True)

    # create grid names
    graticule_names_p = graticule_names(in_raster, block_width, block_height, output_dir)

    dfs = []

    # create grids
    for i, stride_width in enumerate(stride_widths):
        stride_height = stride_heights[i]
        df_tmp, gdf_tmp = raster.graticule(in_raster, block_width, block_height, graticule_names_p[i],
                                           stride=(stride_width, stride_height))
        # left-right edges (we don't want to tile the whole area)
        if i == 4:
            df_tmp, gdf_tmp = filter_out_middle_tiles(df_tmp, gdf_tmp, False,
                                                      graticule_names_p[i])

        # top bottom edges (we don't want to tile the whole area)
        elif i == 5:
            df_tmp, gdf_tmp = filter_out_middle_tiles(df_tmp, gdf_tmp, True,
                                                      graticule_names_p[i])
        else:
            None

        df_tmp["dataset"] = \
        graticule_names_p[i].stem.split(in_raster.stem + "-")[-1]
        dfs.append(df_tmp)

    for i, df in enumerate(dfs):
        # tile from dataframe
        raster.tile_from_dataframe(df, output_dir, block_width, block_height)
        stride_name = graticule_names_p[i].stem.split(in_raster.stem + "-")[-1]
        dataset_directory = output_dir / stride_name / "images"
        if is_tta:
            out_shapefile = output_dir / "shp" / (in_raster.stem + "-" + stride_name + "-boulder-predictions-" +
                                                  scores_str + "-" + min_size_test_str + "-" + max_size_test_str + "-" + config_version + "-tta.shp")
        else:
            out_shapefile = output_dir / "shp" / (in_raster.stem + "-" + stride_name + "-boulder-predictions-" +
                                                  scores_str + "-" + min_size_test_str + "-" + max_size_test_str + "-" + config_version + ".shp")

        if out_shapefile.is_file():
            print("...Predictions for " + out_shapefile.stem + " already exists...")
        else:
            # make predictions
            if is_tta:
                print("...Making tta predictions for " + out_shapefile.stem + "...")
                predict_tta(config_file, model_weights, device, dataset_directory,
                                  out_shapefile, search_tif_pattern,
                                  scores_thresh_test, nms_thresh_test, min_size_test,
                                  max_size_test, pre_nms_topk_test,
                                  post_nms_topk_test, detections_per_image)
            else:
                print("...Making predictions for " + out_shapefile.stem + "...")
                predict(config_file, model_weights, device, dataset_directory,
                                  out_shapefile, search_tif_pattern,
                                  scores_thresh_test, nms_thresh_test, min_size_test,
                                  max_size_test, pre_nms_topk_test,
                                  post_nms_topk_test, detections_per_image)

    return (graticule_names_p)

def picking_predictions_at_centres(in_raster, distance_p, block_width, block_height, graticule_names_p,
                                   scores_str, min_size_test_str,  max_size_test_str, config_version, output_dir, is_tta):
    '''
    This is the second approach.
    '''
    # This covers the first fours
    clipped_boulders = []
    ROI_restricted = []
    res = raster_metadata.get_resolution(in_raster)[0]

    # suffix
    if is_tta:
        suff = "-tta.shp"
    else:
        suff = ".shp"

    print("...Stichting multiple predictions together...")

    for i, graticule_name_p in enumerate(graticule_names_p[:4]):
        stride_name = graticule_name_p.stem.split(in_raster.stem + "-")[-1]
        gdf_graticule = gpd.read_file(graticule_name_p)
        gdf_graticule_restricted = searching_area(gdf_graticule, block_width, block_height, distance_p, res)
        ROI_restricted.append(gdf_graticule_restricted)
        boulder_predictions_p = output_dir / "shp" / (in_raster.stem + "-" + stride_name + "-boulder-predictions-" +
                                              scores_str + "-" + min_size_test_str + "-" + max_size_test_str + "-" + config_version + suff)
        gdf_boulders = gpd.read_file(boulder_predictions_p) 
        clipped_boulders.append(spatial_selection(gdf_boulders, gdf_graticule_restricted))

    # Union of covered
    gdf_union_tmp = gpd.GeoDataFrame(pd.concat(ROI_restricted), crs=ROI_restricted[0].crs)
    gdf_union = gpd.GeoDataFrame(geometry=gpd.GeoSeries(gdf_union_tmp.geometry.unary_union), crs=ROI_restricted[0].crs)

    # difference
    gdf_left_right = gpd.read_file(graticule_names_p[4])
    gdf_top_bottom = gpd.read_file(graticule_names_p[5])
    ROI_restricted_left_right = gpd.GeoSeries(gdf_left_right.unary_union, crs=gdf_left_right.crs).difference(gdf_union).explode(index_parts=True)
    ROI_restricted_left_right.index = np.arange(ROI_restricted_left_right.shape[0]).astype('int')
    ROI_restricted_top_bottom = gpd.GeoSeries(gdf_top_bottom.unary_union, crs=gdf_top_bottom.crs).difference(gdf_union).explode(index_parts=True)
    ROI_restricted_top_bottom.index = np.arange(ROI_restricted_top_bottom.shape[0]).astype('int')

    # Stride (0, block_height/2) - Index 4
    stride_name = graticule_names_p[4].stem.split(in_raster.stem + "-")[-1]
    boulder_predictions_p = output_dir / "shp" / (in_raster.stem + "-" + stride_name + "-boulder-predictions-" +
                                                  scores_str + "-" + min_size_test_str + "-" + max_size_test_str + "-" + config_version + suff)
    gdf_boulders = gpd.read_file(boulder_predictions_p)
    clipped_boulders.append(spatial_selection(gdf_boulders, gpd.GeoDataFrame(geometry=ROI_restricted_left_right, crs=ROI_restricted_left_right.crs)))

    # Stride (block_width/2, 0) - Index 5
    stride_name = graticule_names_p[5].stem.split(in_raster.stem + "-")[-1]
    boulder_predictions_p = output_dir / "shp" / (in_raster.stem + "-" + stride_name + "-boulder-predictions-" +
                                                  scores_str + "-" + min_size_test_str + "-" + max_size_test_str + "-" + config_version + suff)
    gdf_boulders = gpd.read_file(boulder_predictions_p)
    clipped_boulders.append(spatial_selection(gdf_boulders, gpd.GeoDataFrame(geometry=ROI_restricted_top_bottom, crs=ROI_restricted_top_bottom.crs)))

    # Union of all areas
    gdf_union_tmp = gpd.GeoDataFrame(geometry=pd.concat([gdf_union.geometry, ROI_restricted_left_right.geometry,ROI_restricted_top_bottom.geometry]), crs=ROI_restricted[0].crs)
    gdf_union_all = gpd.GeoDataFrame(geometry=gpd.GeoSeries(gdf_union_tmp.unary_union, crs=ROI_restricted[0].crs), crs=ROI_restricted[0].crs)
    gdf_tmp = gpd.GeoSeries(gpd.read_file(graticule_names_p[0]).unary_union, crs=ROI_restricted[0].crs)

    # top left corner
    gdf_diff = gdf_tmp.difference(gdf_union_all)

    # get predictions from no-stride
    stride_name = graticule_names_p[0].stem.split(in_raster.stem + "-")[-1]
    boulder_predictions_p = output_dir / "shp" / (in_raster.stem + "-" + stride_name + "-boulder-predictions-" +
                                                  scores_str + "-" + min_size_test_str + "-" + max_size_test_str + "-" + config_version + suff)
    gdf_boulders = gpd.read_file(boulder_predictions_p)
    clipped_boulders.append(spatial_selection(gdf_boulders, gpd.GeoDataFrame(geometry=gdf_diff, crs=ROI_restricted[0].crs)))

    # concatenate boulder predictions (without filtering)
    gdf_conc = gpd.GeoDataFrame(pd.concat(clipped_boulders), crs=ROI_restricted[0].crs)
    gdf_conc.boulder_id = np.arange(gdf_conc.shape[0]).astype('int')
    gdf_conc.index = gdf_conc.boulder_id.values
    boulder_predictions_conc = output_dir / "shp" / (in_raster.stem + "-boulder-predictions-w-duplicates" + "-" +
                                                     scores_str + "-" + min_size_test_str + "-" + max_size_test_str + "-" + config_version + suff)
    gdf_conc.to_file(boulder_predictions_conc)

    return gdf_conc


def predictions_stitching_filtering(in_raster, config_file, model_weights,
                                    device, search_tif_pattern, distance_p,
                                    block_width, block_height, output_dir, is_tta=True,
                                    scores_thresh_test=0.10,
                                    nms_thresh_test=0.30,
                                    min_size_test=512, max_size_test=512,
                                    pre_nms_topk_test=2000,
                                    post_nms_topk_test=1000,
                                    detections_per_image=2000):
    # predictions
    graticule_names_p = predictions(in_raster, config_file, model_weights, device,
                search_tif_pattern, block_width, block_height, output_dir, is_tta, scores_thresh_test,
                nms_thresh_test, min_size_test, max_size_test, pre_nms_topk_test,
                post_nms_topk_test, detections_per_image)

    scores_str = "scores" + str(int(scores_thresh_test * 100)).zfill(3)
    config_version = [i for i in config_file.stem.split("-") if i.startswith("v0")][0]  # name dependent which is not good...
    min_size_test_str = "minsize" + str(int(min_size_test)).zfill(4)
    max_size_test_str = "maxsize" + str(int(max_size_test)).zfill(4)

    # suffix is_tta
    if is_tta:
        suff = "-tta.shp"
    else:
        suff = ".shp"

    # Only selecting predictions at the centre (include overlapping values) - Slow for large number of boulders
    gdf_conc = picking_predictions_at_centres(in_raster, distance_p, block_width,
                                              block_height, graticule_names_p, scores_str,
                                              min_size_test_str,  max_size_test_str, config_version, output_dir, is_tta)

    # Removal of duplicates with Non Maximum Suppression - Slow for large number of boulders (use torchvision.ops.batched_nms?)
    gdf_final = nms(gdf_conc, nms_thresh_test)

    boulder_predictions_final = output_dir / "shp" / (in_raster.stem + "-" + "boulder-predictions" + "-" +
                                scores_str + "-" + min_size_test_str + "-" + max_size_test_str + "-" + config_version + suff)

    gdf_final.to_file(boulder_predictions_final)

    return gdf_final


def nms(gdf, nms_thresh_test):
    """
    This is not working when index is not linearly increasing...
    Args:
        gdf:
        nms_thresh_test:

    Returns:

    """
    #print("...Removing duplicated boulders with non-maximum suppression...")
    gdf_copy = gdf.copy()

    bboxes = gdf_copy.geometry.bounds.to_numpy()
    bboxes_torch = torch.from_numpy(bboxes)

    scores = gdf_copy.scores.to_numpy()
    scores_torch = torch.from_numpy(scores)
    #idxs_torch = torch.from_numpy(np.zeros(gdf_copy.shape[0]).astype("int"))

    nms_filtered = torchvision.ops.nms(boxes=bboxes_torch, scores=scores_torch, iou_threshold=nms_thresh_test)
    #nms_filtered = torchvision.ops.batched_nms(boxes=bboxes_torch, scores=scores_torch, idxs=idxs_torch, iou_threshold=nms_thresh_test)
    gdf_nms = gdf_copy[gdf_copy.index.isin(nms_filtered.numpy())]
    return gdf_nms


def spatial_selection(gdf, ROI):
    '''
    Spatial selection based on centroid of features.
    '''
    gdf_copy = gdf.copy()
    gdf_copy["geometry"] = gdf.geometry.centroid
    idx_boulder = gpd.overlay(gdf_copy, ROI, how="intersection").boulder_id.values
    gdf_selection = gdf[gdf.boulder_id.isin(idx_boulder)]
    return gdf_selection


def filter_out_middle_tiles(df, gdf, top, graticule_name_p):
    '''
    In some cases (for tile at the left, right, top and bottom, we do not want to run predictions on all tiles.
    This function removes central tiles and only keep tiles located at the edge.
    '''

    graticule_name_p = Path(graticule_name_p)

    df_copy = df.copy()
    gdf_copy = gdf.copy()
    gdf_bounds = gdf.geometry.bounds
    gdf_bounds["tile_id"] = gdf.tile_id.values

    if top:
        tile_id_edge = list(gdf_bounds.tile_id[gdf_bounds.maxy ==
                                               gdf_copy.geometry.total_bounds[
                                                   -1]].values) + list(
            gdf_bounds.tile_id[gdf_bounds.miny == gdf_copy.geometry.total_bounds[1]].values)
    else:
        tile_id_edge = list(gdf_bounds.tile_id[gdf_bounds.maxx ==
                                               gdf_copy.geometry.total_bounds[
                                                   -2]].values) + list(
            gdf_bounds.tile_id[gdf_bounds.minx == gdf_copy.geometry.total_bounds[0]].values)

    gdf_corner = gdf_copy[gdf_copy.tile_id.isin(tile_id_edge)]
    df_corner = df_copy[df_copy.tile_id.isin(tile_id_edge)]

    gdf_corner.to_file(graticule_name_p)
    df_corner.to_pickle(graticule_name_p.parent / graticule_name_p.stem.replace("shp", "pkl"))

    return (df_corner, gdf_corner)


def graticule_names(in_raster, block_width, block_height, output_dir):
    '''
    Generate names of graticules based on block_width, block_height and strides computed from them.
    '''
    filenames = []

    # create strides from block_width and block_height
    stride_widths = [0, int(block_width / 2), int(block_width / 2), block_width,
                     0, int(block_width / 2)]
    stride_heights = [0, int(block_height / 2), block_height,
                      int(block_height / 2), int(block_height / 2), 0]

    for i, stride_width in enumerate(stride_widths):
        stride_height = stride_heights[i]
        filename = (in_raster.stem + "-tiles-" + str(block_width).zfill(
            3) + "x" +
                    str(block_height).zfill(3) + "px-stride-" +
                    str(stride_width).zfill(3) + "-" +
                    str(stride_height).zfill(3) + ".shp")
        filenames.append(output_dir / "shp" / filename)

    return (filenames)


def searching_area(gdf, block_width, block_height, distance_p, res):
    gdf_copy = gdf.copy()
    gdf_copy["geometry"] = gdf_copy.geometry.centroid.buffer(
        (block_width * distance_p * res) / 2.0).envelope
    return gdf_copy

def merging_two_predictions(in_shp_512, in_shp_1024, in_raster, thresh_test_512, thresh_test_1024, nms_thresh, npixels):
    res = raster_metadata.get_resolution(in_raster)[0]
    gdf_512 = gpd.read_file(in_shp_512)
    gdf_512 = gdf_512[gdf_512.scores >= thresh_test_512]
    gdf_512 = gdf_512[gdf_512.geometry.area >= ((res * res) * npixels)] # need at least 20 pixels, 22.4676
    gdf_1024 = gpd.read_file(in_shp_1024)
    gdf_1024 = gdf_1024[gdf_1024.scores >= thresh_test_1024]
    gdf_1024 = gdf_1024[gdf_1024.geometry.area >= ((res * res) * npixels)] # need at least 10-20 pixels
    gdf_conc = gpd.GeoDataFrame(pd.concat([gdf_512, gdf_1024], ignore_index=True)) # if index is repeated the nms will not work!
    gdf_final = nms(gdf_conc, nms_thresh)
    gdf_final.to_file(in_shp_1024.with_name(in_shp_1024.stem.split("-")[0] + "-boulder-predictions-merged-results.shp"))
    return gdf_final
