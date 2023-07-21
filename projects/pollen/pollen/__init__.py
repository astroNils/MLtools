from .config import add_config
from .dataset_mapper import (AlbumentMapper_polygon, AlbumentMapper_bitmask)
from .evaluator import BoulderEvaluator

from .solver import (
    build_optimizer_sgd, build_optimizer_adam, build_lr_scheduler
)

# this allows to do
# from MLtools.projects.bouldering import (add_config, AlbumentMapper_polygon,
# AlbumentMapper_bitmask, BoulderEvaluator, build_optimizer_sgd,
# build_optimizer_adam, build_lr_scheduler)