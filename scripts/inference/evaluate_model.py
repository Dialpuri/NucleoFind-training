import subprocess
import tensorflow as tf
import os
import onnxruntime as rt
import gemmi 
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import zscore
import matplotlib.pyplot as plt
import random

_SIZE = 32
_SPACING = 0.7

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

def _interpolate(
    grid: gemmi.FloatGrid, translation: gemmi.Fractional, rotation: gemmi.Mat33
) -> np.ndarray:
    translation = grid.unit_cell.orthogonalize(translation)
    scale = gemmi.Mat33([[_SPACING, 0, 0], [0, _SPACING, 0], [0, 0, _SPACING]])
    transform = gemmi.Transform(scale.multiply(rotation), translation)
    values = np.zeros((_SIZE, _SIZE, _SIZE), dtype=np.float32)
    grid.interpolate_values(values, transform)
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

def plot(a, b, c, name, epoch):
    num_points = 32
    x_values = np.linspace(0, 31, num_points)
    y_values = np.linspace(0, 31, num_points)
    z_values = np.linspace(0, 31, num_points)
    x_data, y_data, z_data = np.meshgrid(x_values, y_values, z_values)


    # Flatten the data for scatter plot
    x_data_flat = x_data.flatten()
    y_data_flat = y_data.flatten()
    z_data_flat = z_data.flatten()
    w_data_flat = [a.flatten(), b.flatten(), c.flatten()]

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), subplot_kw={'projection': '3d'})
    titles = ["Real", "Target", "Predicted"]

    # Loop for each subplot
    for i, ax in enumerate(axes):
        # Scatter plot with color mapping based on the 4th dimension
        scatter = ax.scatter(x_data_flat, y_data_flat, z_data_flat, s=w_data_flat[i], c=w_data_flat[i], cmap='turbo', marker='o', alpha=0.5)

        # Add color bar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.05)
        cbar.set_label('Density')

        # ax.set_title(f'Grid of Data Points in 3D with 4th Dimension (Plot {i+1})')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')  # This line won't throw an error now

    plt.tight_layout()
    plt.savefig(f"plots/{name}_EPOCH={epoch}.png")

def main(): 

    #tensorflow_model_checkpoint = "/jarvis/jordan/NucleoFind-training-pr/models/02-25-2024-17:24:41_phosphate.best.hdf5" # first attempt
    # tensorflow_model_checkpoint = "/jarvis/jordan/NucleoFind-training-pr/models/02-26-2024-19:46:46_phosphate.best.hdf5" # second attempt
    # tensorflow_model_checkpoint = "/jarvis/jordan/NucleoFind-training-pr/models/02-27-2024-16:10:12_phosphate.best.hdf5"
    tensorflow_model_checkpoint = "/jarvis/jordan/NucleoFind-training-pr/models/02-29-2024-20:10:59_base.best.hdf5"
    # tensorflow_model_checkpoint = "/jarvis/jordan/NucleoFind-training-pr/models/03-02-2024-21:04:29_sugar.best.hdf5"

    tensorflow_model_folder = tensorflow_model_checkpoint.replace("models", "models/tf_sm").rstrip(".best.hdf5")
    # onnx_path = tensorflow_model_checkpoint.replace("models", "models/onnx").replace(".best.hdf5", ".onnx")
    onnx_path = "/jarvis/jordan/NucleoFind-training-pr/onnx_models/base.onnx"

    # if not os.path.exists(onnx_path): 
    model = tf.keras.models.load_model(tensorflow_model_checkpoint, custom_objects={ 
                    "sigmoid_focal_crossentropy": sigmoid_focal_crossentropy,
                    "dice_coe": dice_coe,
                    "dice_loss": dice_loss

                },)
    model.save(tensorflow_model_folder)
    subprocess.run(["python", "-m", "tf2onnx.convert", "--saved-model", tensorflow_model_folder, "--output", onnx_path ])
    return
    model1 = tf.keras.models.load_model(tensorflow_model_checkpoint, custom_objects={ 
                        "sigmoid_focal_crossentropy": sigmoid_focal_crossentropy,
                        "dice_coe": dice_coe, 
                        "dice_loss": dice_loss
                    })
    providers = ['CPUExecutionProvider']
    model = rt.InferenceSession(onnx_path, providers=providers)

    dataset_dir = "/jarvis/jordan/NucleoFind-training-pr/unrefined_reserved_maps"

    experimental_maps = {}
    target_maps = {} 
    target = "phosphate"

    for path in os.scandir(dataset_dir):
        exp_map = os.path.join(path.path, f"experimental.map")
        target_map = os.path.join(path.path, f"{target}.map")

        experimental_maps[path.name] = exp_map
        target_maps[path.name] = target_map

    found = False
    while not found:
        pdb_id = random.choice(list(experimental_maps.keys()))

        exp = gemmi.read_ccp4_map(experimental_maps[pdb_id])
        exp_grid = exp.grid
        exp_grid.normalize()
        tar = gemmi.read_ccp4_map(target_maps[pdb_id])

        translation = gemmi.Fractional(*np.random.rand(3))
        rotation = gemmi.Mat33(Rotation.random().as_matrix())

        exp_array = _interpolate(exp_grid, translation, rotation)
        tar_array = _interpolate(tar.grid, translation, rotation)
        exp_array, tar_array = exp_array , np.array(tf.one_hot(np.round(tar_array).reshape(32,32,32), depth=2))

        if not np.sum(tar_array>0):
            continue


        # eval_ = model.evaluate(exp_array.reshape(1,32,32,32,1), tar_array.reshape(1,32,32,32,2), verbose=2)
        # print(eval_)
        input_name = model.get_inputs()[0].name
        # pred_array = model.predict(exp_array.reshape(1,32,32,32,1))
        pred_array = np.array(model.run(None, {input_name: exp_array.reshape(1,32,32,32,1)})).squeeze()
    
        # print(pred_array)
        pred_array = np.array(tf.argmax(pred_array, axis=-1))
        # print(pred_array.shape)
        # pred_array = np.where(pred_array > 0.1, 1, pred_array)

        # for a,b,c in zip(exp_array.flatten(), tar_array.flatten(), pred_array.flatten()):
        #     # if b == 0.0:
        #     #     continue
        #     print(a,b,c)
        
        # print(exp_array.shape, tar_array.shape, pred_array.shape)
        plot(exp_array, np.array(tf.argmax(tar_array, axis=-1)), pred_array, tensorflow_model_checkpoint.split("/")[-1], 2 )
        found = True 


def eval_1hr2(): 
    # tensorflow_model_checkpoint = "/jarvis/jordan/NucleoFind-training-pr/models/02-26-2024-19:46:46_phosphate.best.hdf5" # second attempt
    tensorflow_model_checkpoint = "/jarvis/jordan/NucleoFind-training-pr/models/02-29-2024-20:10:59_base.best.hdf5"

    tensorflow_model_folder = tensorflow_model_checkpoint.replace("models", "models/tf_sm").rstrip(".best.hdf5")
    onnx_path = tensorflow_model_checkpoint.replace("models", "models/onnx").replace(".best.hdf5", "2.onnx")
    providers = ['CPUExecutionProvider']

    model = rt.InferenceSession(onnx_path, providers=providers)
    mtz = gemmi.read_mtz_file("/jarvis/jordan/NucleoFind-training-pr/debug/1hr2.mtz")

    res = mtz.resolution_high()
    spacing = 0.7
    sample_rate = res/spacing
    
    grid = mtz.transform_f_phi_to_map("FWT", "PHWT", sample_rate=sample_rate)
    grid.normalize()

    # translation = gemmi.Fractional(*np.random.rand(3))
    # rotation = gemmi.Mat33(Rotation.random().as_matrix())

    scale = gemmi.Mat33([[0.7, 0, 0], [0, 0.7, 0], [0, 0, 0.7]])
    transform = gemmi.Transform(scale, gemmi.Vec3(45,5,10))
    values = np.zeros((32, 32, 32), dtype=np.float32)
    grid.interpolate_values(values, transform)

    # exp_array = _interpolate(grid, translation, rotation)

    # tar_array = _interpolate(tar.grid, translation, rotation)
    # exp_array, tar_array = exp_array , np.array(tf.one_hot(np.round(tar_array).reshape(32,32,32), depth=2))

    input_name = model.get_inputs()[0].name
    pred_array = np.array(model.run(None, {input_name: values.reshape(1,32,32,32,1)})).squeeze()
    pred_array = np.array(tf.argmax(pred_array, axis=-1))

    plot(values, np.zeros((32,32,32,1)), pred_array, "1hr2_", 2 )


def eval_interpolation(): 
    # tensorflow_model_checkpoint = "/jarvis/jordan/NucleoFind-training-pr/models/02-26-2024-19:46:46_phosphate.best.hdf5" # second attempt
    tensorflow_model_checkpoint = "/jarvis/jordan/NucleoFind-training-pr/models/02-29-2024-20:10:59_base.best.hdf5"
    tensorflow_model_folder = tensorflow_model_checkpoint.replace("models", "models/tf_sm").rstrip(".best.hdf5")
    onnx_path = tensorflow_model_checkpoint.replace("models", "models/onnx").replace(".best.hdf5", "2.onnx")
    providers = ['CPUExecutionProvider']

    model = rt.InferenceSession(onnx_path, providers=providers)
    mtz = gemmi.read_mtz_file("/jarvis/jordan/NucleoFind-training-pr/debug/1hr2.mtz")
    res = mtz.resolution_high()
    grid = mtz.transform_f_phi_to_map("FWT", "PHWT", sample_rate = res/0.7)
    grid.normalize()
    extent = gemmi.find_asu_brick(grid.spacegroup).get_extent()
    extent.maximum = gemmi.Fractional(1, 1, 1)
    extent.minimum = gemmi.Fractional(0, 0, 0)

    grid_spacing = 0.7

    corners = [
        grid.unit_cell.orthogonalize(fractional)
        for fractional in (
            extent.minimum,
            gemmi.Fractional(extent.maximum[0], extent.minimum[1], extent.minimum[2]),
            gemmi.Fractional(extent.minimum[0], extent.maximum[1], extent.minimum[2]),
            gemmi.Fractional(extent.minimum[0], extent.minimum[1], extent.maximum[2]),
            gemmi.Fractional(extent.maximum[0], extent.maximum[1], extent.minimum[2]),
            gemmi.Fractional(extent.maximum[0], extent.minimum[1], extent.maximum[2]),
            gemmi.Fractional(extent.minimum[0], extent.maximum[1], extent.maximum[2]),
            extent.maximum,
        )
    ]
    min_x = min(corner[0] for corner in corners)
    min_y = min(corner[1] for corner in corners)
    min_z = min(corner[2] for corner in corners)
    max_x = max(corner[0] for corner in corners)
    max_y = max(corner[1] for corner in corners)
    max_z = max(corner[2] for corner in corners)

    box = gemmi.PositionBox()
    box.minimum = gemmi.Position(min_x, min_y, min_z)
    box.maximum = gemmi.Position(max_x, max_y, max_z)
    size: gemmi.Position = box.get_size()
    num_x = int(size.x / grid_spacing)
    num_y = int(size.y / grid_spacing)
    num_z = int(size.z / grid_spacing)

    array = np.zeros((num_x, num_y, num_z), dtype=np.float32)
    scale = gemmi.Mat33(
        [[grid_spacing, 0, 0], [0, grid_spacing, 0], [0, 0, grid_spacing]]
    )

    transform: gemmi.Transform = gemmi.Transform(scale, box.minimum)
    grid.interpolate_values(array, transform)
    cell: gemmi.UnitCell = gemmi.UnitCell(size.x, size.y, size.z, 90, 90, 90)
    interpolated_grid = gemmi.FloatGrid(array, cell)

    sub = np.array(interpolated_grid.get_subarray(
                    start=[0,20,20], shape=[32, 32, 32]
                )).reshape((1, 32, 32, 32, 1))

    input_name = model.get_inputs()[0].name
    pred_array = np.array(model.run(None, {input_name: sub})).squeeze()
    pred_array = np.array(tf.argmax(pred_array, axis=-1))

    plot(sub, np.zeros((32,32,32,1)), pred_array, "interp", 2 )

    


if __name__ == "__main__":
    main()
    # eval_interpolation()
    # eval_1hr2()