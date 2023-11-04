import os
import numpy as np
import pandas as pd
import gemmi
from dataclasses import dataclass
from typing import List
import tensorflow as tf
from tqdm.keras import TqdmCallback

import unet
import enum
import argparse


@dataclass
class Names:
    sugar_file: str = "_sugar"
    phosphate_file: str = "_phosphate"
    base_file: str = "_base"
    density_file: str = "_density"

@dataclass
class Params:
    dataset_base_dir: str = "./dataset"
    shape: int = 32

class Types(enum.Enum):
    SUGAR: int = 1
    PHOSPHATE: int = 2
    BASE: int = 3


def sample_generator(dataset: str = "train"):
    datasets = {"train": "data/train.csv", "test": "data/test.csv"}

    df: pd.DataFrame = pd.read_csv(datasets[dataset])
    df: pd.DataFrame = df.astype({"X": "int", "Y": "int", "Z": "int"})
    df_np: np.ndarray = df.to_numpy()

    def get_density(path: str, translation: List[int]) -> np.ndarray:
        assert len(translation) == 3

        map: gemmi.FloatGrid = gemmi.read_ccp4_map(path).grid
        array: np.ndarray = np.array(
            map.get_subarray(
                start=translation, shape=[param.shape, param.shape, param.shape]
            )
        )
        array = array.reshape((param.shape, param.shape, param.shape, 1))
        return array

    def get_atom_density(path: str, translation: List[int]) -> np.ndarray:
        map: gemmi.FloatGrid = gemmi.read_ccp4_map(path).grid
        array: np.ndarray = np.array(
            map.get_subarray(
                start=translation, shape=[param.shape, param.shape, param.shape]
            )
        )
        hot_array = tf.one_hot(array, depth=2)
        return hot_array

    while True:
        for candidate in df_np:
            assert len(candidate) == 4

            pdb_code: str = candidate[0]
            translation: str = candidate[1:4]

            density_path: str = os.path.join(
                param.dataset_base_dir, pdb_code, f"{pdb_code}{Names.density_file}.map"
            )
            raw_density = get_density(density_path, translation)

            if atom_type == Types.BASE:
                base_path: str = os.path.join(
                    param.dataset_base_dir, pdb_code, f"{pdb_code}{Names.base_file}.map"
                )
                base_array = get_atom_density(base_path, translation)
                yield raw_density, base_array

            if atom_type == Types.SUGAR:
                sugar_path: str = os.path.join(
                    param.dataset_base_dir,
                    pdb_code,
                    f"{pdb_code}{Names.sugar_file}.map",
                )
                sugar_array = get_atom_density(sugar_path, translation)
                yield raw_density, sugar_array

            if atom_type == Types.PHOSPHATE:
                phosphate_path: str = os.path.join(
                    param.dataset_base_dir,
                    pdb_code,
                    f"{pdb_code}{Names.phosphate_file}.map",
                )
                phosphate_array = get_atom_density(phosphate_path, translation)
                yield raw_density, phosphate_array


def sigmoid_focal_crossentropy(
    y_true, y_pred, alpha=0.25, gamma=2.0, from_logits: bool = False
):
    if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero.")

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    # Get the cross_entropy for each entry
    ce = tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha:
        alpha = tf.cast(alpha, dtype=y_true.dtype)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    if gamma:
        gamma = tf.cast(gamma, dtype=y_true.dtype)
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    # compute the final loss and return
    return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)


def train(type):
    num_threads: int = 128
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    tf.config.threading.set_inter_op_parallelism_threads(int(num_threads / 2))
    tf.config.threading.set_intra_op_parallelism_threads(int(num_threads / 2))

    _train_gen = sample_generator("train")
    _test_gen = sample_generator("test")

    input = tf.TensorSpec(shape=(32, 32, 32, 1), dtype=tf.float32)
    output = tf.TensorSpec(shape=(32, 32, 32, 2), dtype=tf.int64)

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
        metrics=["accuracy", "categorical_accuracy"],
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
    name: str = type

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

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(
    #     log_dir=f"./logs/{name}", histogram_freq=1, profile_batch=(10, 30)
    # )

    csv_logger = tf.keras.callbacks.CSVLogger(f"{type}.log")

    callbacks_list = [
        checkpoint,
        reduce_lr_on_plat,
        TqdmCallback(verbose=2),
        csv_logger,
        # tensorboard_callback,
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
    param = Params()
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", required=True)
    args = parser.parse_args()

    atom_type = Types[args.type]
    train(args.type)
