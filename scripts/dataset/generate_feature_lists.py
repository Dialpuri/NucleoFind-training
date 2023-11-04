import os
import random
import gemmi
from typing import List, Tuple
import numpy as np
import pandas as pd
import util
import constants
from multiprocessing import Pool
from tqdm import tqdm 


def generate_candidate_position_list(
    base_dir: str, pdb_code: str, threshold: float
) -> List[List[int]]:
    phos_map = os.path.join(base_dir, f"{pdb_code}{constants.PHOSPHATE_SUFFIX}.map")
    input_grid = gemmi.read_ccp4_map(phos_map).grid

    a = input_grid.unit_cell.a
    b = input_grid.unit_cell.b
    c = input_grid.unit_cell.c

    overlap = 16

    box_dimensions = [32, 32, 32]
    total_points = box_dimensions[0] ** 3

    na = (a // overlap) + 1
    nb = (b // overlap) + 1
    nc = (c // overlap) + 1

    translation_list = []

    for x in range(int(na)):
        for y in range(int(nb)):
            for z in range(int(nc)):
                translation_list.append([x * overlap, y * overlap, z * overlap])

    candidate_translations = []

    for translation in translation_list:
        sub_array = np.array(
            input_grid.get_subarray(start=translation, shape=box_dimensions)
        )

        sum = np.sum(sub_array)
        if (sum / total_points) > threshold:
            candidate_translations.append(translation)

    return candidate_translations


def help_file_worker(data_tuple: Tuple[str, str]):
    base_dir, pdb_code = data_tuple

    # Threshold 0.01 = 10 % of cube has positive density
    candidate_translations = generate_candidate_position_list(base_dir, pdb_code, 0.01)

    help_file_path = os.path.join(base_dir, "feature_list.csv")

    with open(help_file_path, "w") as help_file:
        help_file.write("X,Y,Z\n")

        for translation in candidate_translations:
            help_file.write(f"{translation[0]},{translation[1]},{translation[2]}\n")


def generate_help_files():
    # Must be run after map files have been generated
    dataset = util.get_data_dirs(constants.DATASET_DIRECTORY)

    with Pool() as pool:
        r = list(
            tqdm(
                pool.imap(help_file_worker, dataset),
                total=len(dataset),
            )
        )


# 
# SEED PROTEIN FEATURES INTO FEATURE LISTS
# 

def generate_c_alpha_positions(mtz_path: str, pdb_code: str, sample_size: int ): 

    # Need to find positions to add to the help file which will include position of high density but no sugars
    grid_spacing = 0.7
    mtz = gemmi.read_mtz_file(mtz_path)
    input_grid = mtz.transform_f_phi_to_map("FWT", "PHWT")
    input_grid.normalize()
    try:
        structure = util.get_pdb_path(pdb_code)
    except FileNotFoundError:
        print("[FAILED]:", mtz_path, pdb_code)
        return

    box = util.get_bounding_box(input_grid)
    size = box.get_size()
    num_x = -(-int(size.x / grid_spacing) // 16 * 16)
    num_y = -(-int(size.y / grid_spacing) // 16 * 16)
    num_z = -(-int(size.z / grid_spacing) // 16 * 16)
    array = np.zeros((num_x, num_y, num_z), dtype=np.float32)
    scale = gemmi.Mat33(
        [[grid_spacing, 0, 0], [0, grid_spacing, 0], [0, 0, grid_spacing]]
    )
    transform = gemmi.Transform(scale, box.minimum)
    input_grid.interpolate_values(array, transform)

    c_alpha_search = gemmi.NeighborSearch(structure[0], structure.cell, 3)
    c_alpha_atoms = ["CA", "CB"]

    grid_sample_size = 32

    for n_ch, chain in enumerate(structure[0]):
            for n_res, res in enumerate(chain):
                for n_atom, atom in enumerate(res):
                    if atom.name in c_alpha_atoms:
                        c_alpha_search.add_atom(atom, n_ch, n_res, n_atom)

    potential_positions = []

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                index_pos = gemmi.Vec3(i, j, k)
                position = gemmi.Position(transform.apply(index_pos))

                any_protein_backbone = c_alpha_search.find_atoms(position, "\0", radius=1.5)

                if len(any_protein_backbone) > 0: 
                    translatable_position = (i-grid_sample_size/2, j-grid_sample_size/2, k-grid_sample_size/2)
                    potential_positions.append(translatable_position)

    if len(potential_positions) != 0: 
        return random.sample(potential_positions, sample_size)
    return []


def seeder(data: Tuple[str, str]): 
    mtz_file, pdb_code = data
    
    pdb_folder = os.path.join(constants.DATASET_DIRECTORY, pdb_code)
    output_path = os.path.join(pdb_folder, "seeded_feature_list.csv")

    if os.path.isfile(output_path):
        return 

    validated_translation_file = os.path.join(pdb_folder, "feature_list.csv")

    if not os.path.isfile(validated_translation_file):
        return

    df = pd.read_csv(validated_translation_file)

    if len(df) < 10: 
        sample_size = 4
    else:
        sample_size = len(df) // 5

    samples = generate_c_alpha_positions(mtz_path=mtz_file, pdb_code=pdb_code, sample_size=sample_size)

    output_df = pd.concat([df, pd.DataFrame(samples, columns=["X","Y","Z"])])
    output_df.to_csv(output_path, index=False)

def seed_c_alpha_positions(): 
    mtz_list = util.get_mtz_list(constants.DATA_PATH)

    with Pool() as pool:
        r = list(tqdm(pool.imap(seeder, mtz_list), total=len(mtz_list)))
