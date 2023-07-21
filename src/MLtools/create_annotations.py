import json
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import shapely
import sys

sys.path.append("/home/nilscp/GIT/")

from affine import Affine
from tqdm import tqdm
import rastertools_BOULDERING.raster as raster
from pathlib import Path

def rm_tree(pth):
    pth = Path(pth)
    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()

def generate_graticule_from_raster(in_raster, block_width, block_height, global_graticule_name, stride=(0,0)):

    """
    :param in_raster:
    :param block_width:
    :param block_height:
    :param geopackage:
    :return:
    """
    in_raster = Path(in_raster)
    print("...Generate graticule for raster " + in_raster.name +
          " (" + str(block_width) + "x" + str(block_height) + " pixels, stride " + 
          str(stride[0]) + "/" + str(stride[1]) + ")" + "...")

    global_graticule_name = Path(global_graticule_name)
    global_graticule_name = global_graticule_name.absolute()
    pickle_name = global_graticule_name.with_name(global_graticule_name.stem + ".pkl")
    res = raster.get_raster_resolution(in_raster)[0]

    (windows, transforms, bounds) = raster.tile_windows(in_raster, block_width, block_height, stride)

    assert len(bounds) < 100000, "Number of tiles larger than 100,000. Please modify function generate_graticule_from_raster()."

    polygons = [shapely.geometry.box(l, b, r, t) for l, b, r, t in bounds]
    tile_id = [i for i in range(len(bounds))]
    image_id_png = [in_raster.stem + "_" + str(i).zfill(5) + "_image.png" for i in range(len(bounds))]
    raster_name_abs = [in_raster.as_posix() for i in range(len(bounds))]
    raster_name_rel = [in_raster.name for i in range(len(bounds))]
    windows_px = [list(i.flatten()) for i in windows]
    transforms_p = [list(i)[:6] for i in transforms]
    product_id = [in_raster.stem for i in range(len(bounds))]
    crs = raster.get_raster_crs(in_raster).wkt
    crs_l = [crs for i in range(len(bounds))]
    res_l = [res for i in range(len(bounds))]

    df = pd.DataFrame(list(zip(product_id, tile_id, image_id_png,
                               raster_name_abs, raster_name_rel, windows_px,
                               transforms_p, bounds, crs_l, res_l)),
                      columns=['image_id', 'tile_id', 'file_name',
                               'raster_ap', 'raster_rp', 'rwindows',
                               'transform', 'bbox_im', 'coord_sys', 'pix_res'])
    df.to_pickle(pickle_name)
    df_qgis = df[['image_id', 'tile_id', 'file_name']]

    gdf = gpd.GeoDataFrame(df_qgis, geometry=polygons)
    gdf = gdf.set_crs(crs)

    gdf.to_file(global_graticule_name)
    return (df, gdf)


def clip_boulders(boulders_shapefile, selection_tiles_shapefile, min_area_threshold,
                  global_tiles_pickle, out_selection_tiles_pickle, out_selection_boulders_shapefile):
    """
    After a bit of thinking, I would like to:
    1. clip the boulders for the selected graticule(s)
    2. filter out boulders that have an area below the min_area_threshold.

    It does not make sense to include super tiny boulders at the edge of the
    graticule (even if they are actually pretty large, but on another rectangle
    grid).

    PREVIOUS VERSION:
    In an earlier version I tried to only select boulders that have their
    centers within the graticule and have an area larger than the
    min_area_threshold. However, it was missing boulders at the edge that may
    have their centers in another graticule, but have an area larger than the
    min_area_threshold.

    :param boulders_shapefile:
    :param selection_tiles_shapefile:
    :param min_area_threshold:
    :param global_tiles_pickle:
    :param out_selection_tiles_pickle:
    :param out_selection_boulders_shapefile
    :return:
    """
    # making sure all paths are Path(..)
    boulders_shapefile = Path(boulders_shapefile)
    selection_tiles_shapefile = Path(selection_tiles_shapefile)
    out_selection_boulders_shapefile = Path(out_selection_boulders_shapefile)

    # reading of the selection graticule(s)
    gdf_selection_tiles = gpd.read_file(selection_tiles_shapefile)
    gdf_selection_tiles["coord_sys"] = gdf_selection_tiles.crs.to_wkt()

    # reading of the outline of boulders
    gdf_boulders = gpd.read_file(boulders_shapefile)
    gdf_boulders["coord_sys"] = gdf_boulders.crs.to_wkt()

    frames = []
    empty_bbox = []

    # looping through entries (i.e., through rectangle grids)
    for index, row in tqdm(gdf_selection_tiles.iterrows(),
                           total=gdf_selection_tiles.shape[0]):

        # 1. clipping of intersecting boulders with rectangle grid
        bbox = row.geometry
        # for very specific cases, it can return a MultiPolygon
        # if the edge of boulders oscillate between two rectangle grids
        # I need to tackle this kind of problem.
        gdf_clip = gpd.clip(gdf_boulders, mask=bbox, keep_geom_type=True) # this introduce errors if same boulders zig zag in and out of area
        gdf_clip["area"] = gdf_clip.geometry.area

        # 2. filtering
        gdf_clip = gdf_clip[gdf_clip["area"] >= min_area_threshold]

        if gdf_clip.shape[0] > 0:
            # copy everything except geometry (hardcoded, might be source of error)
            # depending on number of columns in the selection grid (be careful)
            # improve this bit in the future.
            for c in ['image_id', 'tile_id', 'file_name', 'coord_sys']: #gdf_selection_tiles.columns[:3]:
                gdf_clip[c] = row[c]
            frames.append(gdf_clip)

        # let's remove rectangles with no boulders at all!
        else:
            empty_bbox.append(index)

    # concatenate the clipping from each rectangle grid into one gdf
    gdf_boulders = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))

    # tackle Multipolygon(s) problem, exploding multiple features
    # only keep if above min_area_threshold
    if gdf_boulders[gdf_boulders.geometry.geom_type == "MultiPolygon"].shape[0] > 0:
        gdf_multipolygon = gdf_boulders[gdf_boulders.geometry.geom_type == "MultiPolygon"]
        id_multipolygon = gdf_multipolygon.index.values
        gdf_explode = gdf_multipolygon.explode(index_parts=True, ignore_index=True)
        gdf_explode["area"] = gdf_explode.geometry.area
        gdf_explode_selection = gdf_explode[gdf_explode["area"] > min_area_threshold]

        # drop multipolygons in original dataset
        gdf_boulders = gdf_boulders.drop(id_multipolygon)
        print(str(len(id_multipolygon)) + " MultiPolygon(s) was/were removed")

        # only concatenate if at least one boulder is larger than min_area_threshold
        if gdf_explode_selection.shape[0] > 0:
            # add exploded multipolygons above min_area_threshold
            frames = [gdf_boulders, gdf_explode_selection]
            gdf_boulders = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))
            print(str(gdf_explode_selection.shape[0]) + " exploded MultiPolygon(s) was/were added")

    # drop graticule(s)/rectangle grid(s) with non-overlapping boulders
    gdf_selection_tiles_updated = gdf_selection_tiles.drop(empty_bbox)

    # read global graticule pickle to get additional geographical
    # information such as the bounds of the grids and ++
    df_global_tiles = pd.read_pickle(global_tiles_pickle)

    # select only rows that corresponds to the selected rectangle grids
    filter1 = df_global_tiles["tile_id"].isin(gdf_selection_tiles_updated["tile_id"].values)
    df_selection_tiles = df_global_tiles[filter1]
    df_selection_tiles.to_pickle(out_selection_tiles_pickle)

    # is it necessary to save? interesting for debugging and visual inspection
    gdf_boulders.to_file(out_selection_boulders_shapefile)
    # I don't want region of mapping to be overwritten anymore
    #gdf_selection_tiles_updated.to_file(selection_tiles_shapefile)  # overwrite

    print("shapefile " + out_selection_boulders_shapefile.as_posix() + " has been generated")
    #print("shapefile " + selection_tiles_shapefile.as_posix() + " has been overwritten (graticules/rectangle grids with no boulder occurences are deleted)")

    return gdf_boulders, gdf_selection_tiles_updated, df_selection_tiles

def merge_dataframes(frames):
    return(pd.concat(frames, ignore_index=True))

def split_per_image(df_selection_tiles, gdf_selection_tiles, split):
    np.random.seed(seed=27)
    train_tiles = []
    validation_tiles = []
    test_tiles = []

    unique_image_id = df_selection_tiles.image_id.unique()

    numpy_split = [np.round(split[0], decimals=2), np.round(split[0] + split[1], decimals=2)]

    for i in unique_image_id:
        df_selection = df_selection_tiles[df_selection_tiles.image_id == i]
        train_tiles_tmp, val_tiles_tmp , test_tiles_tmp = np.split(
            df_selection.sample(frac=1, random_state=27),
            [int(numpy_split[0]*len(df_selection)),
             int(numpy_split[1]*len(df_selection))])

        train_tiles.append(train_tiles_tmp)
        validation_tiles.append(val_tiles_tmp)
        test_tiles.append(test_tiles_tmp)

    df_train_tiles = pd.concat(train_tiles)
    df_train_tiles["dataset"] = "train"
    df_val_tiles = pd.concat(validation_tiles)
    df_val_tiles["dataset"] = "validation"
    df_test_tiles = pd.concat(test_tiles)
    df_test_tiles["dataset"] = "test"

    df_selection_tiles_split = pd.concat([df_train_tiles, df_val_tiles, df_test_tiles])
    df_selection_tiles_split = df_selection_tiles_split.sample(frac=1, random_state=27)

    # re-index gdf
    gdf_selection_tiles_split = gdf_selection_tiles.reindex(df_selection_tiles_split.index)

    return (df_selection_tiles_split, gdf_selection_tiles_split)

def split_global(df_selection_tiles, gdf_selection_tiles_updated, split):
    """
    Shuffle tiles and randomly distributes the selection rectangle grids /
    graticules into a train / validation / test datasets (respecting the
    split values specified in <split>).

    A "dataset" column is added to both the DataFrame and GeoDataFrame.

    Note that the split here is a global split, meaning that regardless of the
    number of rectangle grids per images, it just randomly distributes across
    the train / validation / test datasets. Another function needs to be written
    if a split per image is wanted... TODO...

    :param df_selection_tiles:
    :param gdf_selection_tiles_updated:
    :param split:
    :return:
    """
    #out_shapefile = Path(out_shapefile)

    np.random.seed(seed=27)
    n = df_selection_tiles.shape[0]
    idx_shuffle = np.random.permutation(n)

    training_idx, remaining_idx = np.split(idx_shuffle, [int(split[0] * len(idx_shuffle))])
    split_val = split[1] / (1 - split[0])  # split compare to remaining data
    val_idx, test_idx = np.split(remaining_idx, [int(split_val * len(remaining_idx))])

    df_selection_tiles["dataset"] = "train"
    df_selection_tiles["dataset"].iloc[val_idx] = "validation"
    df_selection_tiles["dataset"].iloc[test_idx] = "test"

    gdf_selection_tiles_updated["dataset"] = "train"
    gdf_selection_tiles_updated["dataset"].iloc[val_idx] = "validation"
    gdf_selection_tiles_updated["dataset"].iloc[test_idx] = "test"

    return (df_selection_tiles, gdf_selection_tiles_updated)


def folder_structure(df, dataset_directory):
    dataset_directory = Path(dataset_directory)
    folders = list(df["dataset"].unique())
    sub_folders = ["images", "labels"]

    for f in folders:
        for s in sub_folders:
            new_folder = dataset_directory / f / s
            Path(new_folder).mkdir(parents=True, exist_ok=True)

def tiling_raster_from_dataframe(df_selection_tiles, dataset_directory, block_width, block_height):

    print("...Tiling raster(s) from dataframe...")

    dataset_directory = Path(dataset_directory)
    folder_structure(df_selection_tiles, dataset_directory) # ensure folders are created
    datasets = df_selection_tiles.dataset.unique()

    nimages = 0
    for d in datasets:
        image_directory = (dataset_directory / d / "images")
        n = len(list(image_directory.glob("*.tif")))
        nimages = nimages + n

    ntiles = df_selection_tiles.shape[0]

    if nimages == ntiles:
        print("Number of tiles == Number of tiles in specified folder(s). No tiling required.")
    # if for some reasons they don't match, it just need to be re-tiled
    # we delete the image directory(ies) just to start from a clean folder
    else:
        for d in datasets:
            image_directory = (dataset_directory / d / "images")
            rm_tree(image_directory)

        # re-creating folder structure
        folder_structure(df_selection_tiles, dataset_directory)

        for index, row in tqdm(df_selection_tiles.iterrows(), total=ntiles):

            # this is only useful within the loop if generating tiling on multiple images
            in_raster = row.raster_ap
            src_profile = raster.get_raster_profile(in_raster)
            win_profile = src_profile
            win_profile["width"] = block_width
            win_profile["height"] = block_height

            arr = raster.read_raster(in_raster=in_raster, bbox=rio.windows.Window(*row.rwindows))
            h, w = arr.squeeze().shape

            if (h, w) != (block_height, block_width):
                arr = np.pad(arr.squeeze(),
                             [(0, block_height - h), (0, block_width - w)],
                             mode='constant', constant_values=0)
                arr = np.expand_dims(arr, axis=0)

            filename_tif = (dataset_directory / row.dataset / "images" / row.file_name.replace(".png", ".tif"))
            filename_png1 = (dataset_directory / row.dataset / "images" / row.file_name)
            win_profile["transform"] = Affine(*row["transform"])

            # generate tif and pngs (1- and 3-bands)
            raster.save_raster(filename_tif, arr, win_profile, is_image=False)
            raster.tiff_to_png(filename_tif, filename_png1)

def selection_boulders(df_selection_boulders, df_selection_tiles):
    dfs = []
    for i in ["train", "validation", "test"]:
        file_name = df_selection_tiles["file_name"][df_selection_tiles["dataset"] == i]
        filter1 = df_selection_boulders["file_name"].isin(file_name)
        dfs.append(df_selection_boulders[filter1])

    # not necessary but make the code easier to understand
    df_selection_boulders_train = dfs[0]
    df_selection_boulders_train["dataset"] = "train"
    df_selection_boulders_validation = dfs[1]
    df_selection_boulders_validation["dataset"] = "validation"
    df_selection_boulders_test = dfs[2]
    df_selection_boulders_test["dataset"] = "test"

    return (df_selection_boulders_train, df_selection_boulders_validation, df_selection_boulders_test)

def image(df_selection_tiles, dataset, dataset_directory, block_width, block_height, out_pickle):

    out_pickle = Path(out_pickle)
    image_annotations = df_selection_tiles.copy()
    image_annotations = image_annotations[image_annotations["dataset"] == dataset]

    # shuffle images (otherwise the images are ordered by tile_id/file_name)
    image_annotations = image_annotations.sample(frac=1).reset_index(drop=True)

    image_annotations["id"] = np.arange(0, image_annotations.shape[0])
    image_annotations["height"] = block_height
    image_annotations["width"] = block_width
    image_annotations["file_name_ap"] = (dataset_directory / image_annotations["dataset"] / "image" / image_annotations["file_name"])

    image_annotations.to_pickle(out_pickle)
    print("pickle " + out_pickle.as_posix() + " has been generated")

    return (image_annotations)

def segmentation(df_selection_boulders, boulder_annotation_image, dataset_directory, out_pickle):

    """
    Note that both bbox and segmentations need to be given as x,y, while all the
    plots/array adopt a height, width meaning if you have a pixel coordinate
    (2,4), this will correspond to index (4,2) in your array.

    :param df_selection_boulders_train:
    :param boulder_annotation_image:
    :param dataset_directory:
    :return:
    """

    out_pickle = Path(out_pickle)

    bbox_xyxy_pixel = []
    bbox_xyxy_image = []
    bbox_xywh_pixel = []
    image_id = []
    area = []
    segmentation_masks = []

    # looping through boulders
    for index, row in tqdm(df_selection_boulders.iterrows(), total= df_selection_boulders.shape[0]):
        bounds = row.geometry.bounds
        bbox_xyxy_image.append(list(bounds))

        # select rectangle grid on which boulder is located
        df_row = boulder_annotation_image[boulder_annotation_image.file_name == row.file_name]

        # generate the name of the tif file on which the boulder is located on
        tif_filename = dataset_directory / df_row.dataset.iloc[0] / "images" / df_row.file_name.iloc[0].replace("png", "tif")

        image_id.append(df_row["id"].iloc[0]) # save image_id

        # get raster information to convert from image to pixel coordinates
        with rio.open(tif_filename) as src:

            # IMPORTANT!
            # height_min becomes height_max when you switch from image
            # coordinates to pixel coordinates
            height_min, width_min = src.index(bounds[0], bounds[1])
            height_max, width_max = src.index(bounds[2], bounds[3])

            bbox_xyxy_pixel.append([width_min, height_max, width_max, height_min])
            bbox_xywh_pixel.append([width_min, height_max, width_max - width_min, height_min - height_max])
            res = src.res[0]
            area.append(row.geometry.area / (res ** 2.0))  # area in pixel coordinates

            # segmentation part
            x_vertices = list(np.array(row.geometry.exterior.xy[0]))
            y_vertices = list(np.array(row.geometry.exterior.xy[1]))
            # need to take the inverse [::-1] as it returns height, width
            segmentation_mask = [src.index(x, y)[::-1] for x, y in zip(x_vertices, y_vertices)]

            # add 0.5 as in the detectron2 balloon example, probably because
            # they want the mask to be centered in the center of pixels
            # I am not doing it here, but we shoud look closely at mask produced
            segmentation_mask = [p for x in segmentation_mask for p in x]  # [p + 0.5 for x in mask_coord for p in x]
            segmentation_masks.append([segmentation_mask])  # list of a list

    # update the columns with all of the information we need
    df_selection_boulders["id"] = np.arange(0, df_selection_boulders.shape[0])  # boulder_id
    df_selection_boulders["image_id"] = image_id
    df_selection_boulders["category_id"] = 0
    column_names = ["image_id", "bbox_xyxy_pixel", "bbox_xywh_pixel", "bbox_xyxy_image", "segmentation_mask", "area"]
    column_values = [image_id, bbox_xyxy_pixel, bbox_xywh_pixel, bbox_xyxy_image, segmentation_masks, area]

    for i, c in enumerate(column_names):
        df_selection_boulders[c] = column_values[i]
    df_selection_boulders["iscrowd"] = 0

    boulder_annotation_segmentation = df_selection_boulders.copy()

    # saving as pickle and shapefile
    boulder_annotation_segmentation.to_pickle(out_pickle)
    gdf_boulder_annotation_segmentation = gpd.GeoDataFrame(boulder_annotation_segmentation.drop(columns=["bbox_xyxy_pixel", "bbox_xywh_pixel", "bbox_xyxy_image", "segmentation_mask", "area"]))
    gdf_boulder_annotation_segmentation["area"] = gdf_boulder_annotation_segmentation.geometry.area
    gdf_boulder_annotation_segmentation.to_file(out_pickle.with_name(out_pickle.name.replace(".pkl", ".shp")))

    return (boulder_annotation_segmentation)

class NpEncoder(json.JSONEncoder):
    """
    Convert numpy integer/floating and ++ to python integer, float and ++
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def create_category_annotation():
    category_name = "boulder"
    category_id = 0
    category_list = []

    category = {"supercategory": category_name,
                "id": category_id,
                "name": category_name}

    category_list.append(category)

    return category_list

def get_coco_json_format():
    # Standard COCO format
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}]
    }

    return coco_format

def dataframes_to_json_coco_format(image_annotations_df, segmentation_annotations_df, out_json):

    out_json = Path(out_json)

    # get "skeleton" of the coco format
    coco_annotations = get_coco_json_format()

    # convert pandas dataframe to dictionary for image annotations
    coco_annotations["images"] = image_annotations_df[["file_name", "height", "width", "id"]].to_dict(orient='records')

    # convert pandas dataframe to dictionary for segmentation annotations
    segmentation_annotations_df_selection = segmentation_annotations_df[["segmentation_mask", "area", "iscrowd",
                                                                         "image_id", "bbox_xywh_pixel", "category_id", "id"]]

    # rename a few columns
    segmentation_annotations_df_selection = segmentation_annotations_df_selection.rename(
        columns={"segmentation_mask": "segmentation",
                 "bbox_xywh_pixel": "bbox"})

    coco_annotations["annotations"] = segmentation_annotations_df_selection.to_dict(orient='records')

    # create boulder category (hard-coded, see function for more information)
    coco_annotations["categories"] = create_category_annotation()

    with open(out_json, "w") as outfile:
        json.dump(coco_annotations, outfile, indent=4)  # cls=NpEncoder

    print(out_json.name + " has been generated")