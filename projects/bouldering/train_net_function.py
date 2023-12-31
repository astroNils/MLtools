import os
import sys
import torch
import detectron2.utils.comm as comm
sys.path.append("/home/nilscp/GIT/")
sys.path.append("/home/nilscp/GIT/MLtools/projects/bouldering/bouldering")


from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import HookBase

# Bouldering project
from config import add_config
from dataset_mapper import (AlbumentMapper_polygon, AlbumentMapper_bitmask)
from evaluator import BoulderEvaluator
from solver import (build_optimizer_sgd, build_optimizer_adam, build_optimizer_adamw, build_lr_scheduler)

class MyTrainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_train_loader(cls, cfg, is_train=True, sampler=None):
        if cfg.INPUT.MASK_FORMAT == "polygon":
            mapper = AlbumentMapper_polygon(cfg, is_train)
        elif cfg.INPUT.MASK_FORMAT == "bitmask":
            mapper = AlbumentMapper_bitmask(cfg, is_train)
        else:
            None
        return build_detection_train_loader(cfg, mapper=mapper, sampler=sampler)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if cfg.INPUT.MASK_FORMAT == "polygon":
            mapper = AlbumentMapper_polygon(cfg, False)
        elif cfg.INPUT.MASK_FORMAT == "bitmask":
            mapper = AlbumentMapper_bitmask(cfg, False)
        else:
            None
        return build_detection_test_loader(
            cfg, dataset_name, mapper=mapper)


    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
        os.makedirs(output_folder, exist_ok=True)
        return BoulderEvaluator(dataset_name, ("segm",), False, output_folder,
                                max_dets_per_image=1000)

    @classmethod
    def build_optimizer(cls, cfg, model):
        if cfg.SOLVER.OPTIMIZER == "SGD":
            opt = build_optimizer_sgd(cfg, model)
        elif cfg.SOLVER.OPTIMIZER == "ADAM":
            opt = build_optimizer_adam(cfg, model)
        elif cfg.SOLVER.OPTIMIZER == "ADAMW":
            opt = build_optimizer_adamw(cfg, model)
        return opt

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST[0]
        self._loader = iter(MyTrainer.build_train_loader(self.cfg, is_train=False))

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

def train(config_file, config_file_complete, augmentation_file, min_area_npixels, optimizer_n, scheduler_mode="triangular"):

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:', torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:',
              torch.cuda.get_device_properties(0).total_memory / 1e9)

    device = "cuda" if use_cuda else "cpu"
    print("Device: ", device)

    # read augmentations
    cfg = get_cfg()
    add_config(cfg, augmentation_file, min_area_npixels, optimizer_n, scheduler_mode)
    cfg.merge_from_file(config_file)

    # Save complete config file
    with open(config_file_complete, "w") as f:
        f.write(cfg.dump())

    cfg.MODEL.DEVICE = device

    # training
    trainer = MyTrainer(cfg)
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])
    # swap the order of PeriodicWriter and ValidationLoss
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    trainer.train()

