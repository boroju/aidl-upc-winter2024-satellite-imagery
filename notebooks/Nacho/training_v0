import os
import re
import tempfile
import glob
import argparse
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from PIL import Image
from typing import Any, Optional, Union, cast
from typing import Iterable, List
from einops import rearrange
import multiprocessing as mp

# computer vision libraries
import kornia.augmentation as K
from kornia.constants import DataKey, Resample

# Deep Learning libraries
import torch
import torch.nn as nn
import torchdata
import torchvision.models.segmentation
import torchvision.transforms as tf

from sklearn.model_selection import train_test_split
from pickle import dump

import torchgeo
from torchgeo.datasets import RasterDataset, unbind_samples, stack_samples, L8Biome, random_bbox_assignment
from torchgeo.samplers import RandomBatchGeoSampler, RandomGeoSampler, GridGeoSampler, Units
from torch.utils.data import DataLoader
from torchgeo.datamodules import GeoDataModule
from torchgeo.transforms import AugmentationSequential
from torchgeo.samplers.utils import _to_tuple
from torchgeo.datasets.utils import (
    BoundingBox,
    #DatasetNotFoundError,
    #RGBBandsMissingError,
    download_url,
    extract_archive,
)
from torchgeo.models import ResNet18_Weights
from torchgeo.trainers import SemanticSegmentationTask

import pytorch_lightning as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


# satellite images libraries
import rioxarray
import rasterio as rio


# Check if GPU parallel computing is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

from pathlib import Path
root = Path('/opt/ml/processing/input/test')
assert root.exists()

## Wildfire Dataset

class WildFireDaset(RasterDataset):
    filename_glob = "*_landcover_*.tif"
    # filename_glob = "no_fire_landcover_000.tif"

    # Plotting
    all_bands = ["R", "G", "B", "NDVI"]
    rgb_bands = ["R", "G", "B"]
    classes = {
        "no_wildfire": 0,
        "wildfire": 1,
    }

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image, mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)

        filepaths = cast(list[str], [hit.object for hit in hits])

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        image = self._merge_files(filepaths, query, self.band_indexes)

        mask_filepaths = []
        for filepath in filepaths:
            mask_filepath = filepath.replace('_landcover_', '_mask_')

            if os.path.isfile(filepath):
                pass
            else:
                print(f"{filepath} is not a file or does not exist.")

            if os.path.isfile(mask_filepath):
                pass
            else:
                print(f"{mask_filepath} is not a file or does not exist.")

            mask_filepaths.append(mask_filepath)

        mask = self._merge_files(mask_filepaths, query)

        sample = {
            "crs": self.crs,
            "bbox": query,
            "image": image.float(),
            "mask": mask.long(),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionchanged:: 0.3
           Method now takes a sample dict, not a Tensor. Additionally, possible to
           show subplot titles and/or use a custom suptitle.
        """
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                print("ERROR")
                continue

        image = sample["image"][rgb_indices, :, :].permute(1, 2, 0) / 255.0
        # Stretch to the full range
        image = (image - image.min()) / (image.max() - image.min())

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        ax.imshow(image)
        ax.axis("off")
        if show_titles:
            ax.set_title("Image")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig

    def plot_mask(self, sample):
      mask = sample["mask"]
      fig, ax = plt.subplots()
      ax.imshow(mask.squeeze().numpy())
      ax.axis('off')
      return fig

## Normalize

class DatasetStats(torch.nn.Module):
    # "directory", help="directory to recursively search for files"
    # "suffix", default=".tif", help="file suffix"
    # "num_workers", default=1, help="number of threads"

    def __init__(self,
                 directory: str,
                 suffix: Optional[str] = ".tif",
                 num_workers: Optional[int] = 1
    ):
      super().__init__()

      self.directory = directory
      self.suffix = suffix
      self.num_workers = num_workers

    def compute_img_stats(self, path: str) -> tuple["np.typing.NDArray[np.float32]", int]:
        """Compute the min, max, mean, and std dev of a single image.

        Args:
            path: Path to an image file.

        Returns:
            Min, max, mean, and std dev of the image.
        """
        with rio.open(path) as f:
            out = np.zeros((f.count, 4), dtype=np.float32)
            for band in f.indexes:
                stats = f.statistics(band)
                out[band - 1] = (stats.min, stats.max, stats.mean, stats.std)
        return out, f.width * f.height

    def compute_dataset_stats(self):
      paths = glob.glob(
          os.path.join(self.directory, "**", f"*{self.suffix}"), recursive=True
      )

      if self.num_workers > 0:
          out_tuple, size_tuple = list(
              zip(*thread_map(self.compute_img_stats, paths, max_workers=self.num_workers))
          )
          out = np.array(out_tuple)
          sizes = np.array(size_tuple)
      else:
          out_list = []
          size_list = []
          for path in tqdm(paths):
              out, size = self.compute_img_stats(path)
              out_list.append(out)
              size_list.append(size)
          out = np.array(out_list)
          sizes = np.array(size_list)

      # assert len(np.unique(sizes)) == 1

      minimum = np.amin(out[:, :, 0], axis=0)
      maximum = np.amax(out[:, :, 1], axis=0)

      mu_d = out[:, :, 2]
      mu = np.mean(mu_d, axis=0)
      sigma_d = out[:, :, 3]
      N_d = sizes[0]
      N = len(mu_d) * N_d

      # https://stats.stackexchange.com/a/442050/188076
      sigma = np.sqrt(
          np.sum(sigma_d**2 * (N_d - 1) + N_d * (mu - mu_d) ** 2, axis=0) / (N - 1),
          dtype=np.float32,
      )

      np.set_printoptions(linewidth=2**8)
      print("min:", repr(minimum))
      print("max:", repr(maximum))
      print("mean:", repr(mu))
      print("std:", repr(sigma))
      return {"min": repr(minimum), "max":repr(maximum), "mean":repr(mu), "std":repr(sigma)}

dataset_stats_calc = DatasetStats(directory=root.as_posix(), suffix="***landcover***.tif")
stats = dataset_stats_calc.compute_dataset_stats()

stats["mean"]
mean_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", stats["mean"])
mean = [float(num) for num in mean_numbers]

stats["std"]
std_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", stats["std"])
std = [float(num) for num in std_numbers]

mean = mean[0:3]
std = std[0:3]

print(f"mean:{mean}")
print(f"std:{std}")

## Normalize class

class MyNormalize(torch.nn.Module):
    def __init__(self, mean: List[float], stdev: List[float]):
        super().__init__()

        self.mean = torch.Tensor(mean)[:, None, None]
        self.std = torch.Tensor(stdev)[:, None, None]

    def forward(self, inputs: dict):

        x = inputs["image"][..., : len(self.mean), :, :]

        # if batch
        if inputs["image"].ndim == 4:
            x = (x - self.mean[None, ...]) / self.std[None, ...]

        else:
            x = (x - self.mean) / self.std

        inputs["image"][..., : len(self.mean), :, :] = x

        return inputs

    def revert(self, inputs: dict):
        """
        De-normalize the batch.
        Args:
            inputs (dict): Dictionary with the 'image' key
        """

        x = inputs["image"][..., : len(self.mean), :, :]

        # if batch
        if x.ndim == 4:
            x = inputs["image"][:, : len(self.mean), ...]
            x = x * self.std[None, ...] + self.mean[None, ...]
        else:
            x = x * self.std + self.mean

        inputs["image"][..., : len(self.mean), :, :] = x

        return inputs
    
## Wildfire datamodule

class _Transform(nn.Module):
    """Version of AugmentationSequential designed for samples, not batches."""

    def __init__(self, aug: nn.Module) -> None:
        """Initialize a new _Transform instance.

        Args:
            aug: Augmentation to apply.
        """
        super().__init__()
        self.aug = aug

    def forward(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply the augmentation.

        Args:
            sample: Input sample.

        Returns:
            Augmented sample.
        """
        for key in ["image", "mask"]:
            dtype = sample[key].dtype
            # All inputs must be float
            sample[key] = sample[key].float()
            sample[key] = self.aug(sample[key])
            sample[key] = sample[key].to(dtype)
            # Kornia adds batch dimension
            sample[key] = rearrange(sample[key], "() c h w -> c h w")
        return sample

class WildFireDataModule(GeoDataModule):
    """LightningDataModule implementation for the WildFire dataset.
    """
    def __init__(
        self,
        batch_size: int = 1,
        patch_size: Union[int, tuple[int, int]] = 224,
        length: Optional[int] = None,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new L8BiomeDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
            length: Length of each training epoch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.L8Biome`.
        """
        # This is a rough estimate of how large of a patch we will need to sample in
        # EPSG:3857 in order to guarantee a large enough patch in the local CRS.
        self.original_patch_size = patch_size * 3
        # self.original_patch_size = patch_size
        kwargs["transforms"] = _Transform(
            K.CenterCrop(patch_size)
            )

        super().__init__(
            WildFireDaset,
            batch_size=batch_size,
            patch_size=patch_size,
            length=length,
            num_workers=num_workers,
            **kwargs,
        )

        # If you set self.aug (or self.train_aug, self.val_aug, etc. if you need them to differ) in your data module, they will be automatically applied to the entire mini-batch.

        # self.train_aug = AugmentationSequential(
        #     K.Normalize(mean=self.mean, std=self.std),
        #     # K.RandomResizedCrop(_to_tuple(self.patch_size), scale=(0.6, 1.0)),
        #     # K.RandomVerticalFlip(p=0.5),
        #     # K.RandomHorizontalFlip(p=0.5),
        #     data_keys=["image", "mask"],
        #     extra_args={
        #         DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}
        #     },
        # )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        #for i, v in self.kwargs.items():
        #  print ("    ", i, ": ", v)
        dataset = WildFireDaset(**self.kwargs)
        dataset.transforms = MyNormalize(mean, std)

        generator = torch.Generator().manual_seed(0)

        (self.train_dataset, self.val_dataset, self.test_dataset) = (
           random_bbox_assignment(dataset, [0.7, 0.2, 0.1], generator)
        )

        if stage in ["fit"]:
            self.train_batch_sampler = RandomBatchGeoSampler(
                dataset=self.train_dataset, size=self.patch_size, batch_size=self.batch_size, length=self.length
            )
        if stage in ["fit", "validate"]:
            self.val_sampler = GridGeoSampler(
                dataset=self.val_dataset, size=self.patch_size, stride=self.patch_size
            )
        if stage in ["test"]:
            self.test_sampler = GridGeoSampler(
                dataset=self.test_dataset, size=self.patch_size, stride=self.patch_size
            )

        def train_dataloader(self):
            return DataLoader(self.train_dataset, batch_sampler=self.train_batch_sampler, batch_size=self.batch_size)

        def val_dataloader(self):
            return DataLoader(self.val_dataset, sampler=self.val_sampler, batch_size=self.batch_size)

        def test_dataloader(self):
            return DataLoader(self.test_dataset, sampler=self.test_sampler,  batch_size=self.batch_size)

        #def predict_dataloader(self):
        #    return DataLoader(self.mnist_predict, batch_size=self.batch_size)
        
## Model training

WORKERS = mp.cpu_count()
DEVICE, NUM_DEVICES = ("cuda", torch.cuda.device_count()) if torch.cuda.is_available() else ("cpu", mp.cpu_count())
print(f'Running on {NUM_DEVICES} {DEVICE}(s)')
print(f'Number of workers: {WORKERS}(s)')

PATCH_SIZE=512
BATCH_SIZE=12
SAMPLE_SIZE=4000

data_module = WildFireDataModule(paths=root.as_posix(),  patch_size = PATCH_SIZE, batch_size=BATCH_SIZE, num_workers=WORKERS, length=SAMPLE_SIZE)
# data_module = WildFireDataModule(paths=root.as_posix(),  patch_size = PATCH_SIZE, batch_size=BATCH_SIZE)

data_module.prepare_data()
data_module.setup(stage='fit')
data_module.setup(stage='test')

accelerator = "gpu" if torch.cuda.is_available() else "cpu"
default_root_dir = os.path.join(tempfile.gettempdir(), "experiments")
print(default_root_dir)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss", dirpath=default_root_dir, save_top_k=1, save_last=True
)
early_stopping_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10)
logger = TensorBoardLogger(save_dir=default_root_dir, name="tutorial_logs")

max_epochs = 30
fast_dev_run = False

task = SemanticSegmentationTask(
    loss="ce",
    model="deeplabv3+",
    backbone="resnet50",
    weights=True,
    in_channels=4,
    num_classes=2,
    lr=1e-4,
    patience=5,
    ignore_index=None
)

trainer = Trainer(
    accelerator=accelerator,
    callbacks=[checkpoint_callback, early_stopping_callback],
    fast_dev_run=fast_dev_run,
    log_every_n_steps=1,
    logger=logger,
    min_epochs=15,
    max_epochs=max_epochs,
)

trainer.fit(model=task, datamodule=data_module)

trainer.test(model=task, datamodule=data_module)
