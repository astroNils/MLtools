import warnings

# I don't like doing that but pd.Index warnings is a pain in the ass
# and can not be fixed. It will probably go away in future updates
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio as rio
import shapely
import sys

sys.path.append("/home/nilscp/GIT/")

from affine import Affine
from tqdm import tqdm
from rastertools import raster
from shapely.geometry import box
from pycocotools import mask
from pathlib import Path


def rm_tree(pth):
    pth = Path(pth)
    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()


def folder_structure(df, dataset_directory):
    dataset_directory = Path(dataset_directory)
    folders = list(df["dataset"].unique())
    sub_folders = ["images", "labels"]

    for f in folders:
        for s in sub_folders:
            new_folder = dataset_directory / f / s
            Path(new_folder).mkdir(parents=True, exist_ok=True)


def generate_graticule_from_raster(in_raster, block_width, block_height,
                                   global_graticule_name, stride=(0, 0)):
    """
    :param in_raster:
    :param block_width:
    :param block_height:
    :param geopackage:
    :return:
    """
    in_raster = Path(in_raster)
    print("...Generate graticule for raster " + in_raster.name +
          " (" + str(block_width) + "x" + str(
        block_height) + " pixels, stride " +
          str(stride[0]) + "/" + str(stride[1]) + ")" + "...")

    global_graticule_name = Path(global_graticule_name)
    global_graticule_name = global_graticule_name.absolute()
    pickle_name = global_graticule_name.with_name(
        global_graticule_name.stem + ".pkl")
    res = raster.get_raster_resolution(in_raster)[0]

    (windows, transforms, bounds) = raster.tile_windows(in_raster, block_width,
                                                        block_height, stride)

    assert len(
        bounds) < 100000, "Number of tiles larger than 100,000. Please modify function generate_graticule_from_raster()."

    polygons = [shapely.geometry.box(l, b, r, t) for l, b, r, t in bounds]
    tile_id = [i for i in range(len(bounds))]
    image_id_png = [in_raster.stem + "_" + str(i).zfill(5) + "_image.png" for i
                    in range(len(bounds))]
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


def tiling_raster_from_dataframe(df_selection_tiles, dataset_directory,
                                 block_width, block_height):
    print("...Tiling original image into small image patches...")

    dataset_directory = Path(dataset_directory)
    folder_structure(df_selection_tiles,
                     dataset_directory)  # ensure folders are created
    datasets = df_selection_tiles.dataset.unique()

    nimages = 0
    for d in datasets:
        image_directory = (dataset_directory / d / "images")
        n = len(list(image_directory.glob("*.tif")))
        nimages = nimages + n

    ntiles = df_selection_tiles.shape[0]

    if nimages == ntiles:
        print(
            "Number of tiles == Number of tiles in specified folder(s). No tiling required.")
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

            arr = raster.read_raster(in_raster=in_raster,
                                     bbox=rio.windows.Window(*row.rwindows))

            # edge cases (in the East, and South, the extent can be beigger than the actual raster)
            # read_raster will then return an array with not the dimension
            h, w = arr.squeeze().shape

            if (h, w) != (block_height, block_width):
                arr = np.pad(arr.squeeze(),
                             [(0, block_height - h), (0, block_width - w)],
                             mode='constant', constant_values=0)
                arr = np.expand_dims(arr, axis=0)

            filename_tif = (
                        dataset_directory / row.dataset / "images" / row.file_name.replace(
                    ".png", ".tif"))
            filename_png1 = (
                        dataset_directory / row.dataset / "images" / row.file_name)
            win_profile["transform"] = Affine(*row["transform"])

            # generate tif and pngs (1- and 3-bands)
            raster.save_raster(filename_tif, arr, win_profile, is_image=False)
            raster.tiff_to_png(filename_tif, filename_png1)


def tiling_boulders_as_shp_from_df(df_selection_tiles, dataset_directory,
                                   resolution_limit):
    print("...Generating one boulder outline shapefile per image patch...")

    dataset_directory = Path(dataset_directory)
    datasets = df_selection_tiles.dataset.unique()

    nshapefiles = 0
    for d in datasets:
        label_directory = (dataset_directory / d / "labels")
        n = len(list(label_directory.glob("*.shp")))
        nshapefiles = nshapefiles + n

    ntiles = df_selection_tiles.shape[0]

    if nshapefiles == ntiles:
        print(
            "Number of tiles == Number of shapefiles in specified folder(s). No tiling of boulders required.")
    # if for some reasons they don't match, it just need to be re-tiled
    # we delete the image directory(ies) just to start from a clean folder
    else:
        for d in datasets:
            label_directory = (dataset_directory / d / "labels")
            rm_tree(label_directory)

        # re-creating folder structure
        folder_structure(df_selection_tiles, dataset_directory)

        for index, row in tqdm(df_selection_tiles.iterrows(), total=ntiles):

            # this is only useful within the loop if generating tiling on multiple images
            in_raster = row.raster_ap
            in_boulders = row.boulder_ap
            bbox = box(*row.bbox_im)
            gdf_boulders = gpd.read_file(in_boulders, bbox=bbox)
            gdf_boulders["id"] = np.arange(gdf_boulders.shape[0]).astype('int')
            gdf_clip = gpd.clip(gdf_boulders, mask=bbox,
                                keep_geom_type=False)  # to clip at edges
            gdf_clip = gdf_clip[gdf_clip.geometry.geom_type == "Polygon"]
            gdf_clip["area"] = gdf_clip.geometry.area
            gdf_clip = gdf_clip[gdf_clip.area > (
                        resolution_limit * row.pix_res) ** 2.0]  # at least 2 px x 2px areal is adviced
            filename_shp = (
                        dataset_directory / row.dataset / "labels" / row.file_name.replace(
                    "_image.png", "_mask.shp"))
            if gdf_clip.shape[0] > 0:
                gdf_clip.to_file(filename_shp)
            else:
                schema = {"geometry": "Polygon",
                          "properties": {"id": "int", "area": "float"}}
                gdf_empty = gpd.GeoDataFrame(geometry=[])
                gdf_empty.to_file(filename_shp, driver='ESRI Shapefile',
                                  schema=schema, crs=row.coord_sys)


def split_per_image(df_selection_tiles, split):
    np.random.seed(seed=27)
    print("...Assigning train/validation/test datasets to tiles...")
    train_tiles = []
    validation_tiles = []
    test_tiles = []

    unique_image_id = df_selection_tiles.image_id.unique()

    numpy_split = [np.round(split[0], decimals=2),
                   np.round(split[0] + split[1], decimals=2)]

    for i in unique_image_id:
        df_selection = df_selection_tiles[df_selection_tiles.image_id == i]
        train_tiles_tmp, val_tiles_tmp, test_tiles_tmp = np.split(
            df_selection.sample(frac=1, random_state=27),
            [int(numpy_split[0] * len(df_selection)),
             int(numpy_split[1] * len(df_selection))])

        train_tiles.append(train_tiles_tmp)
        validation_tiles.append(val_tiles_tmp)
        test_tiles.append(test_tiles_tmp)

    df_train_tiles = pd.concat(train_tiles)
    df_train_tiles["dataset"] = "train"
    df_val_tiles = pd.concat(validation_tiles)
    df_val_tiles["dataset"] = "validation"
    df_test_tiles = pd.concat(test_tiles)
    df_test_tiles["dataset"] = "test"

    df_selection_tiles_split = pd.concat(
        [df_train_tiles, df_val_tiles, df_test_tiles])
    df_selection_tiles_split = df_selection_tiles_split.sample(frac=1,
                                                               random_state=27)

    return (df_selection_tiles_split)


def semantic_segm_mask(image, labels):
    """No filtering including here"""

    image = Path(image)
    labels = Path(labels)
    seg_mask_filename = Path(
        labels.as_posix().replace("_mask.shp", "_segmask.tif"))

    gdf = gpd.read_file(labels)
    # res = raster.get_raster_resolution(image)[0]
    out_meta = raster.get_raster_profile(image)

    with rio.open(image) as src:
        arr = src.read()  # always read as channel, height, width

        try:
            seg_mask, __ = rio.mask.mask(src, gdf.geometry, all_touched=False, invert=False)
            seg_mask_byte = (seg_mask > 0).astype('uint8')
        # if no values are in there...
        except:
            seg_mask_byte = np.zeros_like(arr).astype('uint8')

    raster.save_raster(seg_mask_filename, seg_mask_byte, out_meta, False)


def gen_semantic_segm_mask(df_selection_tiles, dataset_directory):
    print("...Generating semantic segmentation masks...")
    ntiles = df_selection_tiles.shape[0]
    for index, row in tqdm(df_selection_tiles.iterrows(), total=ntiles):
        image = dataset_directory / row.dataset / "images" / row.file_name.replace(
            "_image.png", "_image.tif")
        labels = dataset_directory / row.dataset / "labels" / row.file_name.replace(
            "_image.png", "_mask.shp")
        semantic_segm_mask(image, labels)


def annotations_to_df(df_selection_tiles, dataset_directory, block_width, block_height, add_one, json_out):
    print("...Generating Detectron2 custom dataset from dataframe...")

    ntiles = df_selection_tiles.shape[0]

    df_json = pd.DataFrame([])
    df_json["file_name"] = df_selection_tiles.file_name
    df_json["height"] = block_height
    df_json["width"] = block_width
    df_json["image_id"] = np.arange(df_json.shape[0]).astype('int')
    df_json["dataset"] = df_selection_tiles.dataset
    annotations = []

    for index, row in tqdm(df_selection_tiles.iterrows(), total=ntiles):

        rle_mask = []
        bbox_xyxy = []

        df_annotations = pd.DataFrame([])

        image = dataset_directory / row.dataset / "images" / row.file_name.replace(
            "_image.png", "_image.tif")
        labels = dataset_directory / row.dataset / "labels" / row.file_name.replace(
            "_image.png", "_mask.shp")

        gdf = gpd.read_file(labels)

        with rio.open(image) as src:
            arr = src.read()  # always read as channel, height, width
            masks = np.zeros((gdf.shape[0], arr.shape[1], arr.shape[2])).astype(
                'uint8')

            # https://rasterio.readthedocs.io/en/stable/api/rasterio.mask.html
            for i, row_gdf in gdf.iterrows():
                out, tt = rio.mask.mask(src, [row_gdf.geometry],
                                        all_touched=False, invert=False)
                masks[i, :, :] = (out[0] > 0).astype('uint8')
                bbox_xyxy.append(bbox_numpy(out[0] > 0, add_one))

            # Then I have to convert the masks (as rle...) bit mask...
            for m in masks:
                rle_mask.append(mask.encode(np.asarray(m, order="F")))

        df_annotations["bbox"] = bbox_xyxy
        df_annotations["bbox_mode"] = 0
        df_annotations["category_id"] = 0
        df_annotations["segmentation"] = rle_mask

        annotations.append(df_annotations.to_dict(orient="records"))

    df_json["annotations"] = annotations
    df_json.to_json(json_out, orient="records", indent=2)

    return (df_json)


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

def wrapper(mapping_dir, saving_dir, split, block_width, block_height, resolution_limit, out_json):

    """

    Args:
        mapping_dir:
        saving_dir:
        split:
        block_width:
        block_height:
        out_json:

    Returns:

    Example:
    mapping_dir = Path("/home/nilscp/tmp/earth/")
    split = (0.80, 0.15, 0.05)
    block_width = 500
    block_height = 500
    saving_dir = Path("/home/nilscp/tmp/mask-test/")
    out_json = Path("/home/nilscp/tmp/mask-test/custom-dataset.json")

    wrapper(mapping_dir, saving_dir, split, block_width, block_height, out_json)
    """

    mapping_dir = Path(mapping_dir)
    saving_dir = Path(saving_dir)

    ROMs = list(mapping_dir.rglob("*ROM.shp"))
    ROMs = [ROM for ROM in ROMs if "/inputs/" in ROM.as_posix()]
    pkls = [list(ROM.parent.glob("*.pkl"))[0] for ROM in ROMs]
    rasters = [list((ROM.parent.parent.parent / "raster").glob("*.tif"))[0] for ROM in ROMs]
    boulder_outlines = [list(ROM.parent.glob("*boulder-mapping.shp"))[0] for ROM in ROMs]
    frames = []

    assert len(ROMs) == len(pkls) == len(rasters) == len(boulder_outlines), "Check number of rasters, labels, and pickles are the same."

    for i, ROM in enumerate(ROMs):
        gdf_rom = gpd.read_file(ROM)
        df_pkl = pd.read_pickle(pkls[i])
        df_pkl_selection = df_pkl[df_pkl.tile_id.isin(gdf_rom.tile_id)]
        frames.append(df_pkl_selection)

    df_all_tiles = pd.concat(frames, ignore_index=True)
    df_split = split_per_image(df_all_tiles, split)

    # if needs to change raster paths
    #df_split['raster_ap'] = df_split['raster_ap'].str.replace(
    #    '/media/nilscp/pampa/BOULDERING/completed_mapping/earth/',
    #    '/home/nilscp/tmp/earth/')

    # tiling of the raster
    tiling_raster_from_dataframe(df_split, saving_dir, block_width, block_height)

    test = df_split.raster_ap.values
    boulder_mapping_col = [list((Path(t).parent.parent / "shp" / "inputs").glob("*boulder-mapping.shp"))[0].as_posix() for t in test]
    df_split["boulder_ap"] = boulder_mapping_col

    # tiling boudler mapping (create pairs of image and shapefile),
    # it would be nice to modify generate_graticule_from_raster to include an absolute path to the boulder_mapping (would make life much
    # easier!)
    tiling_boulders_as_shp_from_df(df_split, saving_dir, resolution_limit)
    # 2 is needed because clip can create polygons less than a pixel, which will result in an empty mask

    # generate semantic segmentation mask
    gen_semantic_segm_mask(df_split, saving_dir)

    # get individual mask and converting it to rle....
    df_custom_dataset = annotations_to_df(df_split, saving_dir, 500, 500, out_json)

    return (df_custom_dataset)