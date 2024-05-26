import os
import numpy as np
import pandas as pd
import gemmi
from dataclasses import dataclass
from typing import List
import tensorflow as tf
from tqdm.keras import TqdmCallback
from scipy.spatial.transform import Rotation
from scipy.stats import zscore
import itertools
import unet
import enum
import argparse
import random
from datetime import datetime
from loss import sigmoid_focal_crossentropy

class Types(enum.Enum):
    sugar: int = 1
    phosphate: int = 2
    base: int = 3


class DataSources(enum.Enum): 
    __order__ = "na_only p_na" # p_na_unref_none p_na_unref_half"
    na_only = 1
    p_na = 2
    # p_na_unref_none = 3
    # p_na_unref_half = 4

_SIZE = 32
_SPACING = 0.7

_DATA_MAP = {
        "train": {             
            "DataSources.na_only": "/vault/NucleoFind-training-pr/train_maps/na_only",
            "DataSources.p_na": "training/maps/p_na",
            # DataSources.p_na_unref_none: "training/maps/p_na_unref_none",
            # DataSources.p_na_unref_half: "training/maps/p_na_unref_half"
        },
        "test": {             
            "DataSources.na_only": "/vault/NucleoFind-training-pr/test_maps/na_only",
            "DataSources.p_na": "training/maps/p_na",
            # DataSources.p_na_unref_none: "training/maps/p_na_unref_none",
            # DataSources.p_na_unref_half: "training/maps/p_na_unref_half"
        }
    }

def sample_generator(mode: str, target: str):
    experimental_maps = {}
    target_maps = {} 

    for source in DataSources: 
        source = str(source)
        experimental_maps.setdefault(source, {})
        target_maps.setdefault(source, {})

        for path in os.scandir(_DATA_MAP[mode][source]):
            exp_map = os.path.join(path.path, f"experimental.map")
            target_map = os.path.join(path.path, f"{target}.map")

            experimental_maps[source][path.name] = gemmi.read_ccp4_map(exp_map)
            target_maps[source][path.name] = gemmi.read_ccp4_map(target_map)

    for source in itertools.cycle(DataSources):
        source = str(source)
        pdb_id = random.choice(list(experimental_maps[source].keys()))

        exp = experimental_maps[source][pdb_id]
        tar = target_maps[source][pdb_id]

        translation = gemmi.Fractional(*np.random.rand(3))
        rotation = gemmi.Mat33(Rotation.random().as_matrix())

        exp_array = _interpolate(exp.grid, translation, rotation)
        tar_array = _interpolate(tar.grid, translation, rotation)

        if np.sum(tar_array > 0):
            yield zscore(exp_array), tar_array


def _interpolate(
    grid: gemmi.FloatGrid, translation: gemmi.Fractional, rotation: gemmi.Mat33
) -> np.ndarray:
    translation = grid.unit_cell.orthogonalize(translation)
    scale = gemmi.Mat33([[_SPACING, 0, 0], [0, _SPACING, 0], [0, 0, _SPACING]])
    transform = gemmi.Transform(scale.multiply(rotation), translation)
    values = np.zeros((_SIZE, _SIZE, _SIZE), dtype=np.float32)
    grid.interpolate_values(values, transform)
    return values[..., np.newaxis]



def train(type):
    num_threads: int = 128
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    tf.config.threading.set_inter_op_parallelism_threads(int(num_threads / 2))
    tf.config.threading.set_intra_op_parallelism_threads(int(num_threads / 2))

    _train_gen = sample_generator("train", type)
    _test_gen = sample_generator("test", type)

    input = tf.TensorSpec(shape=(32, 32, 32, 1), dtype=tf.float32)
    output = tf.TensorSpec(shape=(32, 32, 32, 1), dtype=tf.int64)

    train_dataset = tf.data.Dataset.from_generator(
        lambda: _train_gen, output_signature=(input, output)
    )
    test_dataset = tf.data.Dataset.from_generator(
        lambda: _test_gen, output_signature=(input, output)
    )

    model = unet.binary_model()

    loss = sigmoid_focal_crossentropy

    optimiser = tf.keras.optimizers.Adam(learning_rate=1e-5)

    model.compile(
        optimizer=optimiser,
        loss=loss,
        metrics=["accuracy"],
    )

    reduce_lr_on_plat = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.8,
        patience=5,
        verbose=1,
        mode="auto",
        cooldown=5,
        min_lr=1e-7,
    )
    epochs: int = 100
    batch_size: int = 8
    steps_per_epoch: int = 10000
    validation_steps: int = 1000
    name: str = f"{datetime.now().strftime('%m-%d-%Y-%H:%M:%S')}_{type}"

    weight_path: str = f"models/{name}.best.hdf5"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        weight_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
        save_weights_only=False,
    )

    train_dataset = train_dataset.repeat(epochs).batch(batch_size=batch_size)

    test_dataset = test_dataset.repeat(epochs).batch(batch_size=batch_size)

    csv_logger = tf.keras.callbacks.CSVLogger(f"/vault/NucleoFind-training-pr/logs/{datetime.now().strftime('%m-%d-%Y-%H:%M:%S')}_{type}.log")

    callbacks_list = [
        checkpoint,
        reduce_lr_on_plat,
        TqdmCallback(verbose=2),
        csv_logger,
    ]

    model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        verbose=0,
        use_multiprocessing=True,
    )

    model.save(f"models/{name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", required=True)
    args = parser.parse_args()

    atom_type = Types[args.type]
    train(args.type)
    
