import gemmi
import util
from multiprocessing import Pool
from tqdm import tqdm
import logging
from typing import Tuple
import constants
import numpy as np
import os


def _initialise_neighbour_search(structure: gemmi.Structure, radius: int = 1.5):
    neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, 1).populate()
    sugar_neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, radius)
    phosphate_neigbour_search = gemmi.NeighborSearch(
        structure[0], structure.cell, radius
    )
    base_neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, radius)

    sugar_atoms = constants.get_sugar_atoms()
    phosphate_atoms = constants.get_phosphate_atoms()
    base_atoms = constants.get_base_atoms()

    for n_ch, chain in enumerate(structure[0]):
        for n_res, res in enumerate(chain):
            for n_atom, atom in enumerate(res):
                if atom.name in sugar_atoms:
                    sugar_neigbour_search.add_atom(atom, n_ch, n_res, n_atom)
                elif atom.name in phosphate_atoms:
                    phosphate_neigbour_search.add_atom(atom, n_ch, n_res, n_atom)
                elif atom.name in base_atoms:
                    base_neigbour_search.add_atom(atom, n_ch, n_res, n_atom)

    return (
        neigbour_search,
        sugar_neigbour_search,
        phosphate_neigbour_search,
        base_neigbour_search,
    )


def save_map(array: np.ndarray, pdb_code: str, output_dir: str, suffix: str):
    density_grid = gemmi.FloatGrid(array)
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = density_grid
    ccp4.grid.unit_cell.set(array.shape[0], array.shape[1], array.shape[2], 90, 90, 90)
    ccp4.grid.spacegroup = gemmi.SpaceGroup("P1")
    ccp4.update_ccp4_header()
    density_path = os.path.join(output_dir, f"{pdb_code}{suffix}.map")
    ccp4.write_ccp4_map(density_path)


def generate_class_files(
    mtz_path: str, pdb_code: str, base_dir: str, radius: int = 1.5
):
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

    (
        _,
        sugar_neigbour_search,
        phosphate_neigbour_search,
        base_neigbour_search,
    ) = _initialise_neighbour_search(structure)

    sugar_map = np.zeros(array.shape, dtype=np.float32)
    phosphate_map = np.zeros(array.shape, dtype=np.float32)
    base_map = np.zeros(array.shape, dtype=np.float32)

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                index_pos = gemmi.Vec3(i, j, k)
                position = gemmi.Position(transform.apply(index_pos))

                any_bases = base_neigbour_search.find_atoms(
                    position, "\0", radius=radius
                )
                any_sugars = sugar_neigbour_search.find_atoms(
                    position, "\0", radius=radius
                )
                any_phosphate = phosphate_neigbour_search.find_atoms(
                    position, "\0", radius=radius
                )

                base_mask = 1.0 if len(any_bases) > 1 else 0.0
                sugar_mask = 1.0 if len(any_sugars) > 1 else 0.0
                phosphate_mask = 1.0 if len(any_phosphate) > 1 else 0.0

                sugar_map[i][j][k] = sugar_mask
                phosphate_map[i][j][k] = phosphate_mask
                base_map[i][j][k] = base_mask

    output_dir = os.path.join(base_dir, pdb_code)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_map(array, pdb_code=pdb_code, output_dir=output_dir, suffix=constants.DENSITY_SUFFIX)
    save_map(sugar_map, pdb_code=pdb_code, output_dir=output_dir, suffix=constants.SUGAR_SUFFIX)
    save_map(
        phosphate_map, pdb_code=pdb_code, output_dir=output_dir, suffix=constants.PHOSPHATE_SUFFIX
    )
    save_map(base_map, pdb_code=pdb_code, output_dir=output_dir, suffix=constants.BASE_SUFFIX)


def map_worker(data: Tuple[str, str]):
    mtz_file, pdb_code = data

    generate_class_files(mtz_file, pdb_code, constants.DATASET_DIRECTORY, radius=1.5)


def generate_map_files():
    logging.info("Generating Map Files")
    mtz_list = util.get_mtz_path_list(constants.DATA_PATH)

    with Pool() as pool:
        r = list(tqdm(pool.imap_unordered(map_worker, mtz_list), total=len(mtz_list)))


if __name__ == "__main__":
    generate_map_files()
