import os
import numpy as np
import pandas as pd
import gemmi
from dataclasses import dataclass
from typing import List
import tensorflow as tf
from tqdm.keras import TqdmCallback
from tqdm import tqdm 
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
    __order__ = "na_only protein_na protein_na_unref_none"
    na_only = 1
    protein_na = 2
    protein_na_unref_none = 3


_SIZE = 32
_SPACING = 0.7

_DATA_MAP = {
        "train": {             
            "DataSources.na_only": "/jarvis/jordan/NucleoFind-training-pr/train_maps/na_only",
            "DataSources.protein_na": "/jarvis/jordan/NucleoFind-training-pr/train_maps/protein_na",
            "DataSources.protein_na_unref_none": "/jarvis/jordan/NucleoFind-training-pr/train_maps/protein_na_unref_none",
            # DataSources.p_na_unref_half: "training/maps/p_na_unref_half"
        },
        "test": {             
            "DataSources.na_only": "/jarvis/jordan/NucleoFind-training-pr/test_maps/na_only",
            "DataSources.protein_na": "/jarvis/jordan/NucleoFind-training-pr/test_maps/protein_na",
            "DataSources.protein_na_unref_none": "/jarvis/jordan/NucleoFind-training-pr/test_maps/protein_na_unref_none",
            # DataSources.p_na_unref_half: "training/maps/p_na_unref_half"
        }
    }

def sample_generator(mode: str, target: str, restrict):
    experimental_maps = {}
    target_maps = {} 
    count = 0 

    for source in DataSources: 
        source = str(source)
        experimental_maps.setdefault(source, {})
        target_maps.setdefault(source, {})

        for path in tqdm(os.scandir(_DATA_MAP[mode][source]), total=len(os.listdir(_DATA_MAP[mode][source]))):
            exp_map = os.path.join(path.path, "experimental.map")
            target_map = os.path.join(path.path, f"{target}.map")

            experimental_maps[source][path.name] = exp_map
            target_maps[source][path.name] = target_map

    for source in itertools.cycle(DataSources):
        source = str(source)
        pdb_id = random.choice(list(experimental_maps[source].keys()))

        exp = gemmi.read_ccp4_map(experimental_maps[source][pdb_id])
        exp_grid = exp.grid
        exp_grid.normalize()
        tar = gemmi.read_ccp4_map(target_maps[source][pdb_id])

        translation = gemmi.Fractional(*np.random.rand(3))
        rotation = gemmi.Mat33(Rotation.random().as_matrix())

        exp_array = _interpolate(exp_grid, translation, rotation)
        tar_array = _interpolate(tar.grid, translation, rotation, True)
        
        if restrict:
            if np.sum(tar_array > 0):
                yield exp_array, tf.one_hot(np.round(tar_array), depth=2)
        else:
            yield exp_array, tf.one_hot(np.round(tar_array), depth=2)


def _interpolate(
    grid: gemmi.FloatGrid, translation: gemmi.Fractional, rotation: gemmi.Mat33, squeeze: bool = False
) -> np.ndarray:
    translation = grid.unit_cell.orthogonalize(translation)
    scale = gemmi.Mat33([[_SPACING, 0, 0], [0, _SPACING, 0], [0, 0, _SPACING]])
    transform = gemmi.Transform(scale.multiply(rotation), translation)
    values = np.zeros((_SIZE, _SIZE, _SIZE), dtype=np.float32)
    grid.interpolate_values(values, transform)
    if squeeze:
        return values
    return values[..., np.newaxis]

def dice_coe(y_true,y_pred, loss_type='jaccard', smooth=1.):

    y_true_f = tf.reshape(y_true,[-1])
    y_pred_f = tf.reshape(y_pred,[-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    if loss_type == 'jaccard':
        union = tf.reduce_sum(tf.square(y_pred_f)) + tf.reduce_sum(tf.square(y_true_f))

    elif loss_type == 'sorensen':
        union = tf.reduce_sum(y_pred_f) + tf.reduce_sum(y_true_f)

    else:
        raise ValueError("Unknown `loss_type`: %s" % loss_type)

    return (2. * intersection + smooth) / (union + smooth)

def dice_loss(y_true,y_pred, loss_type='jaccard', smooth=1.):

    y_true_f = tf.cast(tf.reshape(y_true,[-1]),tf.float32)
    y_pred_f =tf.cast(tf.reshape(y_pred,[-1]),tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    if loss_type == 'jaccard':
        union = tf.reduce_sum(tf.square(y_pred_f)) + tf.reduce_sum(tf.square(y_true_f))

    elif loss_type == 'sorensen':
        union = tf.reduce_sum(y_pred_f) + tf.reduce_sum(y_true_f)

    else:
        raise ValueError("Unknown `loss_type`: %s" % loss_type)

    return (1-(2. * intersection + smooth) / (union + smooth))

def train(type, restrict, epochs):
    num_threads: int = 128
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    tf.config.threading.set_inter_op_parallelism_threads(int(num_threads / 2))
    tf.config.threading.set_intra_op_parallelism_threads(int(num_threads / 2))

    _train_gen = sample_generator("train", type, restrict)
    _test_gen = sample_generator("test", type, restrict)

    input = tf.TensorSpec(shape=(32, 32, 32, 1), dtype=tf.float32)
    output = tf.TensorSpec(shape=(32, 32, 32, 2), dtype=tf.int64)

    train_dataset = tf.data.Dataset.from_generator(
        lambda: _train_gen, output_signature=(input, output)
    )
    test_dataset = tf.data.Dataset.from_generator(
        lambda: _test_gen, output_signature=(input, output)
    )

    model = unet.binary_model2()

    loss = sigmoid_focal_crossentropy

    optimiser = tf.keras.optimizers.Adam(learning_rate=1e-3)

    checkpoint_path = "/jarvis/jordan/NucleoFind-training-pr/models/03-01-2024-22:11:41_sugar.best.hdf5"
    model = tf.keras.models.load_model(checkpoint_path, custom_objects={ 
                        "sigmoid_focal_crossentropy": sigmoid_focal_crossentropy,
                        "dice_coe": dice_coe, 
                        "dice_loss": dice_loss
                    })

    model.compile(
        optimizer=optimiser,
        loss=loss,
        metrics=["accuracy", dice_coe],
    )

    reduce_lr_on_plat = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.8,
        patience=2,
        verbose=1,
        mode="auto",
        cooldown=5,
        min_lr=1e-7,
    )
    # epochs: int = 1000
    batch_size: int = 8
    steps_per_epoch: int = 1000
    validation_steps: int = 100
    name: str = f"{datetime.now().strftime('%m-%d-%Y-%H:%M:%S')}_{type}"
    print(f"Starting with {epochs} epochs")

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

    csv_logger = tf.keras.callbacks.CSVLogger(f"/jarvis/jordan/NucleoFind-training-pr/logs/{datetime.now().strftime('%m-%d-%Y-%H:%M:%S')}_{type}.log")

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
    parser.add_argument("-r", "--restrict", action='store_true' )
    parser.add_argument("-e", "--epochs", default=130 )

    args = parser.parse_args()

    atom_type = Types[args.type]
    train(args.type, args.restrict, int(args.epochs))
    
