import albumentations as A
import geopandas as gpd
import numpy as np
import rasterio as rio
import pandas as pd
import torch
import torchvision
import sys

from pathlib import Path
from PIL import Image
from rasterio import features
from shapely.geometry import box
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import detection_utils as utils

sys.path.append("/home/nilscp/GIT/")
from MLtools import create_annotations
from rastertools import raster

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

def default_predictions(in_raster, config_file, model_weights, device, search_tif_pattern,
                        block_width, block_height, output_dir, scores_thresh_test=0.10, nms_thresh_test=0.30,
                        min_size_test=512, max_size_test=512, pre_nms_topk_test=2000, post_nms_topk_test=1000,
                        detections_per_image=2000):

    # maybe have to get rid of output_dir?

    output_dir = Path(output_dir)
    in_raster = Path(in_raster)
    model_weights = Path(model_weights)
    config_file = Path(config_file)
    config_version = [i for i in config_file.stem.split("-") if i.startswith("v0")][0] # name dependent which is not good...
    scores_str = "scores" + str(int(scores_thresh_test*100)).zfill(3)

    # create automatically paths
    (output_dir / "shp").mkdir(parents=True, exist_ok=True)
    graticule_no_stride_p = output_dir / "shp" / (in_raster.stem + "-global-tiles-no-stride.shp")
    graticule_with_stride_p = output_dir / "shp" / (in_raster.stem + "-global-tiles-w-stride.shp")
    graticule_top_bottom_p = output_dir / "shp" / (in_raster.stem + "-global-tiles-top-bottom.shp")
    graticule_left_right_p = output_dir / "shp" / (in_raster.stem + "-global-tiles-left-right.shp")

    # no stride
    dataset_directory = output_dir / "inference-no-stride" / "images"
    out_shapefile = output_dir / "shp" / (in_raster.stem + "-predictions-no-stride-" + scores_str + "-" + config_version + ".shp")
    if out_shapefile.is_file():
        print(out_shapefile.as_posix() + " already exists. Delete file if it needs to be recomputed... ")
    else:
        (df_no_stride, gdf_no_stride) = create_annotations.generate_graticule_from_raster(in_raster, block_width, block_height, graticule_no_stride_p, stride=(0, 0))
        df_no_stride["dataset"] = "inference-no-stride"
        create_annotations.tiling_raster_from_dataframe(df_no_stride, output_dir, block_width, block_height) # only run it if ROM is different other
        predict(config_file, model_weights, device, dataset_directory, out_shapefile, search_tif_pattern, scores_thresh_test, nms_thresh_test,
                        min_size_test, max_size_test, pre_nms_topk_test, post_nms_topk_test,
                        detections_per_image)

    # with stride
    dataset_directory = output_dir / "inference-w-stride" / "images"
    out_shapefile = output_dir / "shp" / (in_raster.stem + "-predictions-w-stride-"  + scores_str + "-" + config_version + ".shp")
    if out_shapefile.is_file():
        print(out_shapefile.as_posix() + " already exists. Delete file if it needs to be recomputed... ")
    else:
        (df_w_stride, gdf_w_stride) = create_annotations.generate_graticule_from_raster(in_raster, block_width, block_height, graticule_with_stride_p, stride=(250, 250))
        df_w_stride["dataset"] = "inference-w-stride"
        create_annotations.tiling_raster_from_dataframe(df_w_stride, output_dir, block_width, block_height)
        predict(config_file, model_weights, device, dataset_directory, out_shapefile, search_tif_pattern, scores_thresh_test, nms_thresh_test,
                        min_size_test, max_size_test, pre_nms_topk_test, post_nms_topk_test,
                        detections_per_image)

    # top bottom
    dataset_directory = output_dir / "inference-top-bottom" / "images"
    out_shapefile = output_dir / "shp" / (in_raster.stem + "-predictions-top-bottom-"  + scores_str + "-" + config_version + ".shp")
    if out_shapefile.is_file():
        print(out_shapefile.as_posix() + " already exists. Delete file if it needs to be recomputed... ")
    else:
        (df3, gdf3) = create_annotations.generate_graticule_from_raster(in_raster, block_width, block_height, graticule_top_bottom_p, stride=(250, 0))
        gdf_bounds = gdf3.geometry.bounds
        gdf_bounds["tile_id"] = gdf3.tile_id.values
        tile_id_edge = list(gdf_bounds.tile_id[gdf_bounds.maxy == gdf3.geometry.total_bounds[-1]].values) + list(
            gdf_bounds.tile_id[gdf_bounds.miny == gdf3.geometry.total_bounds[1]].values)
        gdf_test = gdf3[gdf3.tile_id.isin(tile_id_edge)]
        gdf_test.to_file(graticule_top_bottom_p)
        df3 = df3[df3.tile_id.isin(tile_id_edge)]
        df3["dataset"] = "inference-top-bottom"
        create_annotations.tiling_raster_from_dataframe(df3, output_dir, block_width, block_height)
        predict(config_file, model_weights, device, dataset_directory, out_shapefile, search_tif_pattern, scores_thresh_test, nms_thresh_test,
                        min_size_test, max_size_test, pre_nms_topk_test, post_nms_topk_test,
                        detections_per_image)

    # left right
    dataset_directory = output_dir / "inference-left-right" / "images"
    out_shapefile = output_dir / "shp" / (in_raster.stem + "-predictions-left-right-" + scores_str + "-" + config_version + ".shp")
    if out_shapefile.is_file():
        print(out_shapefile.as_posix() + " already exists. Delete file if it needs to be recomputed... ")
    else:
        (df4, gdf4) = create_annotations.generate_graticule_from_raster(in_raster, block_width, block_height, graticule_left_right_p, stride=(0, 250))
        gdf_bounds = gdf4.geometry.bounds
        gdf_bounds["tile_id"] = gdf4.tile_id.values
        tile_id_edge = list(gdf_bounds.tile_id[gdf_bounds.maxx == gdf4.geometry.total_bounds[-2]].values) + list(gdf_bounds.tile_id[gdf_bounds.minx == gdf4.geometry.total_bounds[0]].values)
        gdf_test = gdf4[gdf4.tile_id.isin(tile_id_edge)]
        gdf_test.to_file(graticule_left_right_p)
        df4 = df4[df4.tile_id.isin(tile_id_edge)]
        df4["dataset"] = "inference-left-right"
        create_annotations.tiling_raster_from_dataframe(df4, output_dir, block_width, block_height)
        predict(config_file, model_weights, device, dataset_directory, out_shapefile, search_tif_pattern, scores_thresh_test, nms_thresh_test,
                        min_size_test, max_size_test, pre_nms_topk_test, post_nms_topk_test,
                        detections_per_image)

    # fixing edge issues
    predictions_no_stride = output_dir / "shp" / (in_raster.stem + "-predictions-no-stride-" + scores_str + "-" + config_version + ".shp")
    predictions_with_stride = output_dir / "shp" / (in_raster.stem + "-predictions-w-stride-" + scores_str + "-" + config_version + ".shp")
    predictions_left_right = output_dir / "shp" / (in_raster.stem + "-predictions-left-right-" + scores_str + "-" + config_version + ".shp")
    predictions_top_bottom = output_dir / "shp" / (in_raster.stem + "-predictions-top-bottom-" + scores_str + "-" + config_version + ".shp")

    # fix invalid geometry (not sure if this work in all cases!)
    __ = quickfix_invalid_geometry(predictions_no_stride)
    __ = quickfix_invalid_geometry(predictions_with_stride)
    __ = quickfix_invalid_geometry(predictions_left_right)
    __ = quickfix_invalid_geometry(predictions_top_bottom)

    stem = in_raster.stem + "-" + scores_str + "-" + config_version

    gdf = fix_edge_cases(predictions_no_stride, predictions_with_stride,
                         predictions_top_bottom, predictions_left_right,
                         graticule_no_stride_p, graticule_with_stride_p,
                         output_dir / "shp")

    return gdf


### Different ways to combine predictions of tiles with different striding ###
'''
It is usually much easier to fix edge artifacts in semantic segmentation model 
as you can apply a filter with weight values that is function of the distance 
from the centre of the tile.

There is a probably some literature about it and this part of the code 
should be improved in the future. 

Quick summary of tested techniques (more information can be found below):
 
1. My first try was to only replace boulders at the edge. But after looking at 
the results, I noticed that the predictions close to the edges were not that 
great and could be improved. Also, I have to admit that some parts of the code 
for the first technique is bugged. Many functions were written to remove
duplicates when I could have simply used non-maximum suppression, which is 
available in the torchvision package. I have learned from my own mistakes :)
For example the removal of duplicates produce sometimes a shapefile with larger 
number of boulders than at the start. This technique is no longer in used, so
I am not even bothering in finding the source of those problems. However, I am, 
for now keeping, this code to have an overview of what I have been trying.  

2. The second approach (currently used) is to only select boulders within X% 
from the center of the tile. In order to have predictions everywhere (as it 
only selects the predictions at the centre of the tiles), the numbers of 
striding setups are increased from 4 to 6. If X is taken to be larger 
than 50% (distance_p = 0.5), there will be duplicates in the dataset. I remove
those duplicates with the help of non-maximum suppression (torchvision).

Striding setups:
- Stride (0, 0) - Index 0 in gdfs 
----> Selecting (distance_p * 100) % from centre.
- Stride (block_width/2, block_height/2) - Index 1
----> Selecting (distance_p * 100) % from centre.
- Stride (block_width/2, block_height) - Index 2
----> Selecting (distance_p * 100) % from centre.
- Stride (block_width, block_height/2) - Index 3
----> Selecting (distance_p * 100) % from centre.
- Stride (0, block_height/2) - Index 4
----> Selecting all boulders at the left and rightmost tiles (not covered by 0-3).
- Stride (block_width/2, 0) - Index 5
----> Selecting all boulders at the top and bottommost tiles (not covered by 0-3).

The functions related to each of this approach are gathered below. 
'''

### First approach ###
'''
This is the first approach we have used. Only boulders at the edges are replaced.


Predictions:
- Stride (0, 0) - No stride.
- Stride (block_width/2, block_height/2) - Stride.
- Stride (0, block_height/2) - Stride along the left and right edges of the raster.
- Stride (block_width/2, 0) - Stride along the top and bottom edges of the raster.

1. All predictions in the no-stride (0,0) are first selected. 
2. Predictions crossing the edges of the no-stride tiles are selected.
3. Predictions in the shifted tiles are looked for...
'''


def geometry_for_inference(gra_no_stride, output_filename, output_dir):
    # ---------------------------------------------------------------------------
    # Generating edges as polyline for the center parts of the image
    print(
        "...generating geometries for correcting predictions located at edges...")
    gra_center = gra_no_stride.geometry.bounds
    gra_center["tile_id"] = gra_no_stride.tile_id.values
    a = (gra_center.minx != gra_no_stride.geometry.total_bounds[0]) & (
            gra_center.maxx != gra_no_stride.geometry.total_bounds[2])

    b = (gra_center.miny != gra_no_stride.geometry.total_bounds[1]) & (
            gra_center.maxy != gra_no_stride.geometry.total_bounds[3])

    c = gra_center[a & b]
    tile_id_edge = c.tile_id.values
    d = gra_no_stride[gra_no_stride.tile_id.isin(tile_id_edge)]
    gra_center = d.geometry.boundary
    gra_center = gpd.GeoDataFrame(geometry=gpd.GeoSeries(gra_center))

    # ---------------------------------------------------------------------------
    # Generating edges as polyline for the top and bottom parts of the image
    gra_edge_top_bottom = gra_no_stride.geometry.bounds
    gra_edge_top_bottom["tile_id"] = gra_no_stride.tile_id.values
    tile_id_edge = list(
        gra_edge_top_bottom.tile_id[
            gra_edge_top_bottom.maxy == gra_no_stride.geometry.total_bounds[
                -1]].values) + list(
        gra_edge_top_bottom.tile_id[
            gra_edge_top_bottom.miny == gra_no_stride.geometry.total_bounds[
                1]].values)
    gra_edge_top_bottom = gra_no_stride[
        gra_no_stride.tile_id.isin(tile_id_edge)]
    gra_edge_top_bottom = gra_edge_top_bottom.geometry.boundary
    gra_edge_top_bottom = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries(gra_edge_top_bottom))
    gra_edge_top_bottom = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries(gra_edge_top_bottom.difference(
            box(*gra_no_stride.total_bounds).boundary), crs=gra_no_stride.crs))

    gra_edge_top_bottom = gpd.GeoDataFrame(geometry=gpd.GeoSeries(
        gra_edge_top_bottom.difference(d.geometry.unary_union.boundary),
        crs=gra_no_stride.crs))

    # ---------------------------------------------------------------------------
    # Generating edges as polyline for the left and right parts of the image
    gra_edge_left_right = gra_no_stride.geometry.bounds
    gra_edge_left_right["tile_id"] = gra_no_stride.tile_id.values

    tile_id_edge = list(gra_edge_left_right.tile_id[gra_edge_left_right.maxx ==
                                                    gra_no_stride.geometry.total_bounds[
                                                        -2]].values) + list(
        gra_edge_left_right.tile_id[
            gra_edge_left_right.minx == gra_no_stride.geometry.total_bounds[
                0]].values)
    gra_edge_left_right = gra_no_stride[
        gra_no_stride.tile_id.isin(tile_id_edge)]
    gra_edge_left_right = gra_edge_left_right.geometry.boundary
    gra_edge_left_right = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries(gra_edge_left_right))
    gra_edge_left_right = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries(gra_edge_left_right.difference(
            box(*gra_no_stride.total_bounds).boundary), crs=gra_no_stride.crs))
    gra_edge_left_right = gpd.GeoDataFrame(geometry=gpd.GeoSeries(
        gra_edge_left_right.difference(d.geometry.unary_union.boundary),
        crs=gra_no_stride.crs))

    # ---------------------------------------------------------------------------
    # Updating the two last ones to get the right polylines
    sp = gpd.overlay(gra_edge_top_bottom, gra_edge_left_right,
                     how='intersection', keep_geom_type=False)
    sp = sp[sp.geometry.geom_type == "LineString"]
    sp_bounds = sp.bounds
    gra_line_top_bottom = sp_bounds[
        np.logical_or(sp_bounds.miny == np.min(sp_bounds.miny),
                      sp_bounds.maxy == np.max(sp_bounds.maxy))]
    gra_line_left_right = sp_bounds[
        np.logical_or(sp_bounds.minx == np.min(sp_bounds.minx),
                      sp_bounds.maxx == np.max(sp_bounds.maxx))]

    gra_edge_left_right_final = gpd.overlay(gra_edge_left_right,
                                            sp.loc[gra_line_top_bottom.index],
                                            how="symmetric_difference",
                                            keep_geom_type=False)

    gra_edge_top_bottom_final = gpd.overlay(gra_edge_top_bottom,
                                            sp.loc[gra_line_left_right.index],
                                            how="symmetric_difference",
                                            keep_geom_type=False)

    gra_center.to_file(
        output_dir / output_filename[0])  # "edge-lines-center.shp"
    gra_edge_left_right_final.to_file(
        output_dir / output_filename[1])  # "edge-lines-left-right.shp"
    gra_edge_top_bottom_final.to_file(
        output_dir / output_filename[2])  # "edge-lines-top-bottom.shp"

    return (gra_center, gra_edge_left_right_final, gra_edge_top_bottom_final)


def lines_where_boulders_intersect_edges(gdf_boulders, gdf_lines,
                                         output_filename, output_dir):
    """
    Maybe better to give filename? --> avoid loading the dataset every times..

    :param boulders_shp:
    :param lines_shp:
    :param output_dir:
    :return:
    """
    edge_issues = gpd.overlay(gdf_boulders, gdf_lines, how='intersection',
                              keep_geom_type=False)
    edge_issues.geometry.geom_type.unique()

    # only keep the longest line when a multilinestring is generated
    gdf_MultiLineString = edge_issues[
        edge_issues.geometry.geom_type == 'MultiLineString']
    gdf_MultiLineString = gdf_MultiLineString.explode(index_parts=True)
    gdf_MultiLineString["length"] = gdf_MultiLineString.geometry.length
    gdf_MultiLineString = gdf_MultiLineString.sort_values(
        by=["boulder_id", "length"])
    gdf_MultiLineString = gdf_MultiLineString.drop_duplicates(
        subset="boulder_id", keep='last')

    gdf_LineString = edge_issues[edge_issues.geometry.geom_type == 'LineString']
    gdf_AllLines = gpd.GeoDataFrame(
        pd.concat([gdf_LineString, gdf_MultiLineString], ignore_index=True))
    if gdf_AllLines.shape[0] > 0:
        gdf_AllLines.to_file(output_dir / output_filename)
    else:
        None
    return (gdf_AllLines)


def replace_boulder_intersecting(gdf_boulders_original, gdf_boulders_replace,
                                 gdf_edge_intersections, output_filename,
                                 output_dir):
    """
    VERY IMPORTANT:
    Should I have a flag, only replace if a "hit" is found for the same?
    or should I just expect that in order for a detection to be robust,
    it needs to be detected in multiple predictions?

    :param gdf_boulders_original:
    :param gdf_boulders_replace:
    :param gdf_edge_intersections:
    :param output_filename:
    :return:
    """
    print("...replacing boulders at edge...")
    if gdf_edge_intersections.shape[0] > 0:
        gdf_edge_intersections.boulder_id = gdf_edge_intersections.boulder_id.astype(
            'int')
        idx_boulders_at_edge = gdf_edge_intersections.boulder_id.unique()

        gdf_boulders_original_at_edge = gdf_boulders_original[
            gdf_boulders_original.boulder_id.isin(idx_boulders_at_edge)]
        gdf_boulders_original_not_at_edge = gdf_boulders_original[
            ~(gdf_boulders_original.boulder_id.isin(idx_boulders_at_edge))]

        # Finding intersecting boulders in the replacement GeoDataFrame
        gdf_boulders_replace_intersecting = gpd.overlay(gdf_boulders_replace,
                                                        gdf_edge_intersections,
                                                        how='intersection',
                                                        keep_geom_type=False)

        # if no intersecting boulders found in the replacement GeoDataFrame
        if gdf_boulders_replace_intersecting.shape[0] == 0:
            gdf = gdf_boulders_original
        else:
            idx_boulders_at_edge = gdf_boulders_replace_intersecting.boulder_id_1.unique()  # problem here sometimes... if no intersection send error message...
            gdf_boulders_replace_at_edge = gdf_boulders_replace[
                gdf_boulders_replace.boulder_id.isin(idx_boulders_at_edge)]

            gdf = gpd.GeoDataFrame(pd.concat([gdf_boulders_original_not_at_edge,
                                              gdf_boulders_replace_at_edge],
                                             ignore_index=True))
            gdf.boulder_id = np.arange(gdf.shape[0])
    else:
        gdf = gdf_boulders_original
    gdf.to_file(output_dir / output_filename)
    return (gdf)


def fix_double_edge_cases(gra_no_stride, gra_w_stride):
    hot_spot = gpd.overlay(
        gpd.GeoDataFrame(geometry=gpd.GeoSeries(gra_no_stride.boundary)),
        gpd.GeoDataFrame(geometry=gpd.GeoSeries(gra_w_stride.boundary)),
        how='intersection', keep_geom_type=False)

    # the drop duplicates is extremely slow, so it is better to create two
    # columns (x and y) and drop duplicated based on it.
    hot_spot = hot_spot.explode(index_parts=True, ignore_index=True)
    x = [x.coords[0][0] for x in hot_spot.geometry]
    y = [x.coords[0][1] for x in hot_spot.geometry]
    hot_spot["x"] = x
    hot_spot["y"] = y
    hot_spot = hot_spot.drop_duplicates(subset=['x', 'y'])

    return (hot_spot.drop(columns=["x", "y"]))


# there is a problem with this function, it takes lot of time to run it...
def replace_boulders_at_double_edge(gra_no_stride, gra_w_stride, gdf_no_stride,
                                    gdf_w_stride, gdf_last, output_filename,
                                    output_dir):
    """
    One of the problem is that even if we use with-stride predictions for boulders
    intersecting the edge of the no-stride grid, there are still places where the
    same boulders will remain splited. This is true for intersections between the
    no-stride and with-stride grids. This function merge boulders that touch
    the two intersections (here referred as hot spot) as a solution.

    :param graticule_no_stride:
    :param graticule_with_stride:
    :param gdf_no_stride:
    :param gdf_w_stride:
    :param gdf_last:
    :return:
    """
    print(
        "...replacing boulders that intersects no- and with- stride graticules...")
    # hotspot for errors
    hot_spot = fix_double_edge_cases(gra_no_stride, gra_w_stride)

    # fixing errors
    # I think that the lines below fail if there are no overlaps?
    boulders_at_double_edge_ns = gpd.overlay(gdf_no_stride, hot_spot,
                                             how='intersection',
                                             keep_geom_type=False)
    boulders_at_double_edge_ws = gpd.overlay(gdf_w_stride, hot_spot,
                                             how='intersection',
                                             keep_geom_type=False)
    boulders_at_double_edge_fp = gpd.overlay(gdf_last, hot_spot,
                                             how='intersection',
                                             keep_geom_type=False)

    idx1 = boulders_at_double_edge_ns.boulder_id.unique()
    idx2 = boulders_at_double_edge_ws.boulder_id.unique()
    idx3 = boulders_at_double_edge_fp.boulder_id.unique()

    gdf_tom = gpd.GeoDataFrame(
        pd.concat([gdf_no_stride[gdf_no_stride.boulder_id.isin(idx1)],
                   gdf_w_stride[gdf_w_stride.boulder_id.isin(idx2)],
                   gdf_last[gdf_last.boulder_id.isin(idx3)]],
                  ignore_index=True))

    gdf_double_edge = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries(gdf_tom.geometry.unary_union,
                               crs=gdf_no_stride.crs)).explode(index_parts=True)
    gdf_not_double_edge = gdf_last[~(gdf_last.boulder_id.isin(idx3))]
    gdf = gpd.GeoDataFrame(
        pd.concat([gdf_double_edge, gdf_not_double_edge], ignore_index=True))
    gdf = gdf.set_crs(gdf_no_stride.crs, allow_override=True)
    gdf.boulder_id = np.arange(gdf.shape[0])
    gdf.to_file(output_dir / output_filename)
    return (gdf)


def merging_overlapping_boulders(gdf_final, output_filename, output_dir):
    """
    This function merge all of overlapping features that intersect each other,
    and have more than 50% overlap between the smallest and the largest polygon
    of the overlapping features. This function works with multiple polygons if
    all polygons intersect each other.

    For example if you have polygons A, B and C, A being the largest polygon,
    and that AB, AC and BC intersect each other, all good.

    If B intersects A, but C intersects only B, it can
    (if overlap is more than 50%) potentially result in two merged polygons. So,
    you still may have overlapping features at the end... but it should (hopefully)
    be for a very tiny portion of boulders...

    :param gdf_final:
    :return:
    """
    print("...merging overlapping boulders...")
    overlapping = gpd.overlay(gdf_final, gdf_final, how='intersection',
                              keep_geom_type=False)
    overlapping = overlapping[
        overlapping.boulder_id_1 != overlapping.boulder_id_2]

    # line and points and other stuff do not represent a good overlap
    overlapping = overlapping[
        np.logical_or(overlapping.geometry.geom_type == "Polygon",
                      overlapping.geometry.geom_type == "MultiPolygon")]

    boulder_idx = list(overlapping.boulder_id_1.values)
    gdf_overlap = gdf_final[gdf_final.boulder_id.isin(boulder_idx)]
    gdf_overlap["area"] = gdf_overlap.geometry.area
    gdf_non_overlapping = gdf_final[~(gdf_final.boulder_id.isin(
        boulder_idx))]  # data, the inverse can be taken for non-overlapping

    # overlapping contains combination A and B and B and A (which is the same)
    # we get rid of it by multiplying boulder_id_1 by boulder_id_2
    # it creates an unique combination, which we use to drop duplicates
    # combinations
    overlapping["multi"] = (overlapping.boulder_id_1 + 1) * (
                overlapping.boulder_id_2 + 1)
    overlapping.drop_duplicates(subset="multi", keep='first', inplace=True)
    overlapping = overlapping.drop(columns=['multi'])

    merge_list = []
    geom_of_merged = []

    # looping through potential combination of boulders that can be merged
    for index, row in tqdm(overlapping.iterrows(), total=overlapping.shape[0]):
        gdf_selection = gdf_overlap[
            gdf_overlap.boulder_id.isin([row.boulder_id_1, row.boulder_id_2])]
        gdf_selection = gdf_selection.sort_values(by=["area"])
        value = gdf_selection.iloc[0].geometry.intersection(
            gdf_selection.iloc[1].geometry).area / gdf_selection.iloc[
                    0].geometry.area
        if value > 0.50:
            merge_list.append(True)
            geom_of_merged.append(gdf_selection.geometry.unary_union)
        else:
            merge_list.append(False)
            geom_of_merged.append(0)

    overlapping["is_merged"] = merge_list
    overlapping["geom_merged"] = geom_of_merged
    overlapping["geometry"] = overlapping["geom_merged"]
    overlapping = overlapping.drop(columns=["geom_merged"])

    overlapping_tbm = overlapping[overlapping["is_merged"] == True]
    not_overlapping = overlapping[overlapping["is_merged"] == False]

    idx_not_overlapping = sorted(
        list(not_overlapping.boulder_id_1.unique()) + list(
            not_overlapping.boulder_id_2.unique()))
    gdf_overlap_but_not = gdf_overlap[
        gdf_overlap.boulder_id.isin(idx_not_overlapping)]  # DATA

    # need to drop a few values
    gdf_non_overlapping = gdf_non_overlapping[gdf_non_overlapping.columns[
        gdf_non_overlapping.columns.isin(['geometry', 'boulder_id'])]]
    overlapping_tbm = overlapping_tbm[
        overlapping_tbm.columns[overlapping_tbm.columns.isin(['geometry'])]]
    overlapping_tbm["boulder_id"] = 0
    gdf_overlap_but_not = gdf_overlap_but_not[gdf_overlap_but_not.columns[
        gdf_overlap_but_not.columns.isin(['geometry'])]]
    gdf_overlap_but_not["boulder_id"] = 0

    gdf = gpd.GeoDataFrame(
        pd.concat([gdf_non_overlapping, overlapping_tbm, gdf_overlap_but_not],
                  ignore_index=True))
    gdf = gdf.set_crs(gdf_final.crs, allow_override=True)
    gdf.boulder_id = np.arange(gdf.shape[0])
    gdf.to_file(output_dir / output_filename)
    return (gdf)


def fix_edge_cases(predictions_no_stride, predictions_with_stride,
                   predictions_top_bottom, predictions_left_right,
                   graticule_no_stride, graticule_with_stride, output_dir):
    output_dir = Path(output_dir)

    # retrieve name information
    stem = predictions_no_stride.stem
    scores_name = stem.split("-")[-2]
    version_name = stem.split("-")[-1]
    raster_name = stem.split("-")[0]

    gdf_no_stride = gpd.read_file(predictions_no_stride)
    gdf_w_stride = gpd.read_file(predictions_with_stride)
    gdf_left_right = gpd.read_file(predictions_left_right)
    gdf_top_bottom = gpd.read_file(predictions_top_bottom)

    gra_no_stride = gpd.read_file(graticule_no_stride)
    gra_w_stride = gpd.read_file(graticule_with_stride)

    gdf_no_stride["boulder_id"] = gdf_no_stride["boulder_id"].values.astype(
        'int')
    gdf_w_stride["boulder_id"] = gdf_w_stride["boulder_id"].values.astype('int')
    gdf_left_right["boulder_id"] = gdf_left_right["boulder_id"].values.astype(
        'int')
    gdf_top_bottom["boulder_id"] = gdf_top_bottom["boulder_id"].values.astype(
        'int')

    output_filename = ("edge-lines-center.shp", "edge-lines-left-right.shp",
                       "edge-lines-top-bottom.shp")
    (gra_center, gra_edge_left_right_final,
     gra_edge_top_bottom_final) = geometry_for_inference(gra_no_stride,
                                                         output_filename,
                                                         output_dir)

    lines_center = lines_where_boulders_intersect_edges(gdf_no_stride,
                                                        gra_center,
                                                        'lines-where-boulder-intersects-center.shp',
                                                        output_dir)
    lines_topbottom = lines_where_boulders_intersect_edges(gdf_no_stride,
                                                           gra_edge_top_bottom_final,
                                                           'lines-where-boulder-intersects-top-bottom.shp',
                                                           output_dir)
    lines_leftright = lines_where_boulders_intersect_edges(gdf_no_stride,
                                                           gra_edge_left_right_final,
                                                           'lines-where-boulder-intersects-left-right.shp',
                                                           output_dir)

    preliminary_predictions_p = raster_name + "-preliminary-predictions-" + scores_name + "-" + version_name + ".shp"
    final_predictions_p = raster_name + "-final-predictions-" + scores_name + "-" + version_name + ".shp"
    gdf1 = replace_boulder_intersecting(gdf_no_stride, gdf_w_stride,
                                        lines_center, preliminary_predictions_p,
                                        output_dir)
    gdf2 = replace_boulder_intersecting(gdf1, gdf_top_bottom, lines_topbottom,
                                        preliminary_predictions_p, output_dir)
    gdf3 = replace_boulder_intersecting(gdf2, gdf_left_right, lines_leftright,
                                        preliminary_predictions_p, output_dir)
    gdf_degde = replace_boulders_at_double_edge(gra_no_stride, gra_w_stride,
                                                gdf_no_stride, gdf_w_stride,
                                                gdf3, preliminary_predictions_p,
                                                output_dir)
    gdf_final = merging_overlapping_boulders(gdf_degde, final_predictions_p,
                                             output_dir)
    return gdf_final


def quickfix_invalid_geometry(boulders_shp):
    print("...fixing invalid geometries...")
    gdf_boulders = gpd.read_file(boulders_shp)
    valid_geom_idx = gdf_boulders.geometry.is_valid

    if ~valid_geom_idx.all():
        n = gdf_boulders[valid_geom_idx == False].shape[0]
        print(str(n) + " invalid geometry(ies) detected")
        gdf_valid = gdf_boulders[valid_geom_idx]
        gdf_invalid = gdf_boulders[~valid_geom_idx]
        gdf_invalid["geometry"] = gdf_invalid.geometry.buffer(0)
        gdf_boulders = gpd.GeoDataFrame(
            pd.concat([gdf_valid, gdf_invalid], ignore_index=False))
    else:
        None
    if gdf_boulders.shape[0] > 0:  # if non-empty
        gdf_boulders.to_file(boulders_shp)
    else:
        None
    return (gdf_boulders)


### Second approach ###

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
        df_tmp, gdf_tmp = create_annotations.generate_graticule_from_raster(
            in_raster, block_width, block_height,
            graticule_names_p[i], stride=(stride_width, stride_height))
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
        create_annotations.tiling_raster_from_dataframe(df, output_dir,
                                                        block_width,
                                                        block_height)
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
    res = raster.get_raster_resolution(in_raster)[0]

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
    res = raster.get_raster_resolution(in_raster)[0]
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
