from detectron2.config import CfgNode as CN

__all__ = ["add_config"]

def add_config(cfg, aug_path, min_area_npixels, optimizer_n, scheduler_mode):
    """
    Add config for Bouldering project.
    """

    # Optimizer parameters (ADAM, ADAMW, SGD)
    cfg.SOLVER.OPTIMIZER = optimizer_n

    # Only use for CyclicLR scheduler (hardcoded).
    # Based on 753 images, and image batch of 4.
    cfg.SOLVER.MAX_LR = 0.004
    cfg.SOLVER.STEP_SIZE_UP = 800
    cfg.SOLVER.MODE = scheduler_mode # triangular, triangular2 or exp_range

    # thresholding mask (I don't have any thresholding possibility for polygons)
    cfg.INPUT.MIN_AREA_NPIXELS = min_area_npixels

    # augmentations
    cfg.MODEL.AUGMENTATIONS = CN()
    cfg.MODEL.AUGMENTATIONS.PATH = aug_path