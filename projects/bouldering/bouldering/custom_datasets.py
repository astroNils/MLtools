import pandas as pd
from detectron2.data import MetadataCatalog, DatasetCatalog
from pathlib import Path
def get_boulder_dicts(json_file, img_dir, dataset_type):
    df = pd.read_json(json_file)
    df_copy = df.copy()
    df_copy["file_name"] = img_dir / df.dataset / "images" / df.file_name
    df_copy["file_name"] = df_copy["file_name"].astype(str)
    df_dataset= df_copy[df_copy.dataset == dataset_type]
    df_dataset = df_dataset.drop(columns=["dataset"])
    return (df_dataset.to_dict(orient="records"))


def generate_custom_dataset(json_file, img_dir, dataset_name):

    for d in ["train", "validation", "test"]:
        DatasetCatalog.register(dataset_name + "_" + d, lambda d=d: get_boulder_dicts(json_file, img_dir, d))
        MetadataCatalog.get(dataset_name + "_" + d).set(thing_classes=["boulder"])
    boulder_metadata = MetadataCatalog.get(dataset_name + "_train")

    return boulder_metadata

def bitmask_dataset():
    json_file = Path("/home/nilscp/tmp/mask-test/test.json")
    img_dir = Path("/home/nilscp/tmp/mask-test/")
    dataset_name = "boulder_bitmask_5px"
    generate_custom_dataset(json_file, img_dir, dataset_name)