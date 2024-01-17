import numpy as np
import albumentations as A
import wandb
import yaml
import segmentation_models_pytorch as smp
import torch
import pytorch_lightning as pl
import pandas as pd


from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from rastertools_BOULDERING import raster
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from typing import List, Any, Union
from torch.utils.data import Dataset
from pytorch_lightning import seed_everything
seed_everything(27)


wandb.login()

# initialise the wandb logger and name your wandb project
# the actual runs logs will be saved in the dir variable specified in wandb.init
wandb_logger = WandbLogger(project='bouldering', log_model=False) #False --> checkpoints not saved, True --> at the end of training, "all", each epoch?


# lr_scheduler
# loss_fn (unified focal loss (mixed of the next too, dice loss, focal loss)
# max_num_of_epochs
# include gradient x and y
# reshape_size

def load_aug_dict(filepath: Union[str, Path]) -> Any:
    """
    to generate Albu pipeline --> A.from_dict(load_aug_dict(filepath))
    """
    return(pd.read_json(filepath).to_dict())
def default_scheduler_config(lr_scheduler):
    lr_scheduler_config = {
        # REQUIRED: The scheduler instance
        "scheduler": lr_scheduler,
        # The unit of the scheduler's step size, could also be 'step'.
        # 'epoch' updates the scheduler on epoch end whereas 'step'
        # updates it after a optimizer update.
        "interval": "epoch",
        # How many epochs/steps should pass between calls to
        # `scheduler.step()`. 1 corresponds to updating the learning
        # rate after every epoch/step.
        "frequency": 1,
        # I do not want to use multiple schedulers, so one name should be enough
        "name": "lr",
    }
    return lr_scheduler_config

# I see by not defining the step
class BoulderModel(pl.LightningModule):

    def __init__(self, arch="Unet", encoder_name="resnet34", encoder_weights="imagenet", in_channels=3,
                 out_classes=1, optimizer_name="Adam", augmentations="block1", loss_fn_name="DiceLoss",
                 scheduler_name="CosineAnnealingLR", max_epochs=10,
                 patience_or_cycle_length=5, milestones=False, lr=1e-2, lr_min=1e-5, batch_size=8, **kwargs):
        super().__init__()

        # defining a few variables
        self.train_step_outputs = []
        self.valid_step_outputs = []
        self.test_step_outputs = []

        self.arch = arch
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.optimizer_name = optimizer_name
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.lr = lr
        self.lr_min = lr_min
        self.patience_or_cycle_length = patience_or_cycle_length # in epochs
        self.milestones = milestones # list of epochs where it is decrease by 0.1 [2, 8, 25]
        self.batch_size = batch_size
        self.scheduler_name = scheduler_name
        self.augmentations = augmentations
        self.max_epochs = max_epochs

        self.model = smp.create_model(arch, encoder_name=encoder_name, encoder_weights=encoder_weights,
            in_channels=in_channels, classes=out_classes, **kwargs)

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice (dice loss)
        if loss_fn_name == "DiceLoss":
            self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        elif loss_fn_name == "FocalLoss":
            self.loss_fn = smp.losses.FocalLoss(smp.losses.BINARY_MODE, from_logits=True)
        elif loss_fn_name == "JaccardLoss":
            self.loss_fn = smp.losses.JaccardLoss(smp.losses.BINARY_MODE, from_logits=True)
        elif loss_fn_name == "LovaszLoss":
            self.loss_fn = smp.losses.LovaszLoss(smp.losses.BINARY_MODE, from_logits=True)

        # can be specified to multiple loss functions self.loss_fn1, self.loss_fn2, self.loss_fn3, self.loss_fn4

        # save hyperparameters to self.hparamsm auto-logged by wandb
        self.save_hyperparameters()

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        
        image = batch["image"]
        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)
        # see here for multiple loss functions
        # https://github.com/Lightning-AI/pytorch-lightning/issues/2645

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):

        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # mean of loss 
        loss_mean = torch.stack([x["loss"] for x in outputs]).mean()

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_loss": loss_mean,
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }        
        # see https://lightning.ai/docs/pytorch/stable/extensions/logging.html
        # I am a bit confused by steps/epoch
        # this is saving as multiple steps even if it is
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "train")
        self.train_step_outputs.append(loss)
        return loss            

    def on_train_epoch_end(self):
        outputs = self.train_step_outputs
        self.shared_epoch_end(outputs, "train")
        self.train_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "valid")
        self.valid_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        outputs = self.valid_step_outputs
        self.shared_epoch_end(outputs, "valid")
        self.valid_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, "test")
        self.test_step_outputs.append(loss)
        return loss  

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        self.shared_epoch_end(outputs, "test")
        self.test_step_outputs.clear()

    ### Issues with number of steps and for example some schedulers
    ### https://github.com/Lightning-AI/pytorch-lightning/issues/5449
    ### https://docs.wandb.ai/guides/track/log/logging-faqs
    ### https://lightning.ai/docs/pytorch/stable/extensions/logging.html
    ### Controlling logging frequency
    def configure_optimizers(self):
        ## Optimizer
        if self.optimizer_name == "Adam":
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), lr=self.lr)
        elif self.optimizer_name == "RAdam":
            optimizer = optim.RAdam(self.parameters(), lr=self.lr)
        elif self.optimizer_name == "AdamW":
            optimizer = optim.AdamW(self.parameters(), lr=self.lr)

        ## Scheduler
        ### See https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        if self.scheduler_name:
            if self.scheduler_name == "ReduceLROnPlateau":
                # mode = max if IoU or min if loss
                lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                                    patience=self.patience_or_cycle_length,
                                                                    min_lr=self.lr_min)
                lr_scheduler_config = {
                    # REQUIRED: The scheduler instance
                    "scheduler": lr_scheduler,
                    # The unit of the scheduler's step size, could also be 'step'.
                    # 'epoch' updates the scheduler on epoch end whereas 'step'
                    # updates it after a optimizer update.
                    "interval": "epoch",
                    # How many epochs/steps should pass between calls to
                    # `scheduler.step()`. 1 corresponds to updating the learning
                    # rate after every epoch/step.
                    "frequency": 1,
                    # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                    # valid_per_image_iou
                    "monitor": "valid_loss",
                    # If set to `True`, will enforce that the value specified 'monitor'
                    # is available when the scheduler is updated, thus stopping
                    # training if not found. If set to `False`, it will only produce a warning
                    "strict": True,
                    # I do not want to use multiple so should be good
                    "name": "lr",
                }
            elif self.scheduler_name == "CosineAnnealingLR":
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.patience_or_cycle_length,
                                                                    eta_min=self.lr_min)
                lr_scheduler_config = default_scheduler_config(lr_scheduler)
            elif self.scheduler_name == "MultiStepLR":
                lr_scheduler = optim.lr_scheduler.MultiStepLR(milestones=self.milestones, gamma=0.10)
                lr_scheduler_config = default_scheduler_config(lr_scheduler)


        if self.scheduler_name:
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}
        else:
            return optimizer

# to configurate the scheduler, just add those two lines
#self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=CONFIG['t_max'], eta_min=CONFIG['min_lr'])
#return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler}

def read_yaml(f):
    with open(f, "r") as stream:
        try:
            y = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        return y


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def transpose_compose(to_tensor):
    _transform = [A.Lambda(image=to_tensor, mask=to_tensor)]
    return A.Compose(_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)

class SegmentationDataset(Dataset):
    def __init__(self, images: List[Path], masks: List[Path] = None, transforms=None, preprocessing=None) -> None:
        self.images = images
        self.masks = masks
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.images[idx]
        image = np.tile(np.expand_dims(raster.read(image_path).squeeze(), -1), (1, 1, 3))  # H, W, C
        result = {"image": image}

        if self.masks is not None:
            mask_path = self.masks[idx]
            mask = np.expand_dims(raster.read(mask_path).squeeze(), -1)  # H, W, C
            result["mask"] = mask

        if self.transforms is not None:
            transform = self.transforms(image=image, mask=mask)  # need to include more information in the compose...
            image, mask = transform["image"], transform["mask"]

        if self.preprocessing is not None:
            preproc = self.preprocessing(image=image, mask=mask)
            image, mask = preproc['image'], preproc['mask']

        # Change to # C, H, W
        result["image"] = image
        result["mask"] = mask
        result["filename"] = image_path.name

        return result

# needs to be changed
pre_processed_folder = Path("D:/BOULDERING/data/preprocessing_Jan2024/") # all data
train_folder = Path("D:/BOULDERING/data/preprocessing_Jan2024/train/")
validation_folder = Path("D:/BOULDERING/data/preprocessing_Jan2024/validation/")
test_folder = Path("D:/BOULDERING/data/preprocessing_Jan2024/test/")

train_image_patches = sorted(list(train_folder.rglob("*_image.tif")))
train_lbl_patches = [i.parent.parent / "labels" /  i.name.replace("image", "segmask") for i in train_image_patches]
train_image_patches_pngs = [i.parent.parent / "images" /  i.name.replace(".tif", ".png") for i in train_image_patches]

val_image_patches = sorted(list(validation_folder.rglob("*_image.tif")))
val_lbl_patches = [i.parent.parent / "labels" /  i.name.replace("image", "segmask") for i in val_image_patches]
val_image_patches_pngs = [i.parent.parent / "images" /  i.name.replace(".tif", ".png") for i in val_image_patches]

test_image_patches = sorted(list(test_folder.rglob("*_image.tif")))
test_lbl_patches = [i.parent.parent / "labels" /  i.name.replace("image", "segmask") for i in test_image_patches]
test_image_patches_pngs = [i.parent.parent / "images" /  i.name.replace(".tif", ".png") for i in test_image_patches]

### Augmentations (maybe not the best way of doing it....)

############### Default transform ###############
default_transform = A.LongestMaxSize(max_size=512, interpolation=2, p=1.0)

############### Block 1 ###############
block1 = A.OneOf([A.NoOp(p=1.0),
                  A.Affine(p=1.0, rotate=90.0),
                  A.Affine(p=1.0, rotate=180.0),
                  A.Affine(p=1.0, rotate=270.0),
                  A.HorizontalFlip(p=1.0),
                  A.VerticalFlip(p=1.0),
                  A.Transpose(p=1.0),
                  A.Transpose(p=1.0)], p=1.0)

block1_and_default = A.Compose([default_transform, block1])

############### Block 2 ###############
block2 = A.OneOf([A.CLAHE(p=1.0, clip_limit=(1.0, 4.0), tile_grid_size=(8,8)),
                  A.RandomGamma(p=1.0, gamma_limit=(80, 150)),
                  A.GaussNoise(p=1.0, var_limit=(10.0, 50.0), mean=0, per_channel=False),
                  A.Sharpen(p=1.0),
                  A.Blur(p=1.0, blur_limit=(3,7)),
                  A.NoOp(p=1.0)], p=1.0)

block2_and_default = A.Compose([default_transform, block2])
block12_and_default = A.Compose([default_transform, block1, block2])

############### Block 3 ###############
block3_p1 = A.Compose([A.LongestMaxSize(max_size=[640, 768, 896, 1024, 1152, 1280], interpolation=2, p=1.0),
                       A.CropNonEmptyMaskIfExists(height=512, width=512, p=0.5)], p=1.0)  # 256?
block3_p2 = A.NoOp(p=0.5)

block3 = A.OneOf([block3_p1, block3_p2], p=1.0)

block3_and_default = A.Compose([default_transform, block3])
block13_and_default = A.Compose([default_transform, block1, block3])
block123_and_default = A.Compose([default_transform, block1, block2, block3])

############### Block 4 ###############
block4 = A.OneOf([A.ElasticTransform(p=1.0, alpha=1, sigma=50, alpha_affine=25, interpolation=2, border_mode=0),
                  A.GridDistortion(p=1.0, num_steps=8, distort_limit=0.6, interpolation=2, border_mode=0),
                  A.GridDropout(p=1.0, ratio=0.5, unit_size_min=32, unit_size_max=128, random_offset=True, fill_value=0, mask_fill_value=0), # could be better
                  A.NoOp(p=1.0)], p=1.0)

block1234_and_default = A.Compose([default_transform, block1, block2, block3, block4])
block134_and_default = A.Compose([default_transform, block1, block3, block4])
block4_and_default = A.Compose([default_transform, block4])
block14_and_default = A.Compose([default_transform, block1, block4])

### default config file
p = Path("D:/BOULDERING/nb/segmentation_tutorial/bouldering") # needs to be changed
config_path = p / "default_config.yml"
config = read_yaml(config_path)
running_dir = Path(".").absolute() / "models"
running_dir.mkdir(parents=True, exist_ok=True)
torch.set_float32_matmul_precision('medium')

def main():

    config = read_yaml(config_path) # default values
    # I guess the sweep is overriding
    # dir=<> otherwise default to ./wandb
    run = wandb.init(project="bouldering", config=config, dir=running_dir)

    # init the autoencoder
    model = BoulderModel(**wandb.config)

    # augmentation
    if wandb.config["augmentations"] == "default":
        aug = default_transform
    elif wandb.config["augmentations"] == "block1":
        aug = block1_and_default
    elif wandb.config["augmentations"] == "block2":
        aug = block2_and_default
    elif wandb.config["augmentations"] == "block3":
        aug = block3_and_default
    elif wandb.config["augmentations"] == "block4":
        aug = block4_and_default
    elif wandb.config["augmentations"] == "blocks12":
        aug = block12_and_default
    elif wandb.config["augmentations"] == "blocks13":
        aug = block13_and_default
    elif wandb.config["augmentations"] == "blocks14":
        aug = block14_and_default
    elif wandb.config["augmentations"] == "blocks123":
        aug = block123_and_default
    elif wandb.config["augmentations"] == "blocks134":
        aug = block134_and_default
    elif wandb.config["augmentations"] == "blocks1234":
        aug = block1234_and_default


    # load datasets
    ### Loaders could be loaded in the model too... but it feels a little bit weird...
    ### see https://pytorch-lightning.readthedocs.io/en/1.1.8/introduction_guide.html
    training_dataset = SegmentationDataset(train_image_patches, train_lbl_patches,
                                           transforms=aug,
                                           preprocessing=transpose_compose(
                                               to_tensor))  # let's do the processing in the model...

    # preprocessing=get_preprocessing(preprocessing_fn)

    validation_dataset = SegmentationDataset(val_image_patches, val_lbl_patches,
                                             transforms=default_transform,
                                             preprocessing=transpose_compose(to_tensor))

    test_dataset = SegmentationDataset(test_image_patches, test_lbl_patches,
                                       transforms=default_transform,
                                       preprocessing=transpose_compose(to_tensor))

    # loaders
    train_loader = DataLoader(training_dataset, batch_size=wandb.config["batch_size"], shuffle=True, num_workers=0)
    valid_loader = DataLoader(validation_dataset, batch_size=wandb.config["batch_size"], shuffle=False, num_workers=0)
    #test_loader = DataLoader(test_dataset, batch_size=wandb.config["batch_size"], shuffle=False, num_workers=0)

    # saving of checkpoints
    checkpoint_p = f"models/{run.name}/checkpoints/"
    checkpoint_callback = ModelCheckpoint(monitor='valid_per_image_iou', dirpath=checkpoint_p, 
        filename='bouldering-{epoch:02d}-{valid_per_image_iou:.2f}', save_top_k=3, mode='max')

    lr_monitor = LearningRateMonitor(logging_interval='step') # should be epoch, but it does not show up in wandb

    # check here if want to be saved every_n_steps, can be specified in wandblogger.
    # https://pytorch-lightning.readthedocs.io/en/1.4.9/extensions/generated/pytorch_lightning.callbacks.ModelCheckpoint.html
    
    # pass wandb_logger to the Trainer 
    trainer = pl.Trainer(max_epochs=wandb.config["max_epochs"], callbacks=[checkpoint_callback, lr_monitor], logger=wandb_logger)
    
    # train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader,)

    run.finish()

if __name__ == "__main__":
    main()
