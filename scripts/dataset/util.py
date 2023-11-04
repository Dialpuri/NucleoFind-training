import os
from typing import List, Tuple
import gemmi
import constants


def get_mtz_path_list(directory: str) -> List[Tuple[str, str]]:
    # Returns list of tuples containing mtz_path and pdb_code
    mtz_list = os.listdir(directory)
    return [
        (os.path.join(directory, mtz_file), mtz_file.replace(".mtz", ""))
        for mtz_file in mtz_list
        if ".mtz" in mtz_file
    ]

def get_pdb_path(pdb_code: str) -> str:
    # Return filepath of supplied PDB code from a local PDB archive
    middlefix = pdb_code[1:3]
    pdb_archive = "/vault/pdb"
    filename = f"pdb{pdb_code}.ent"
    return os.path.join(pdb_archive, middlefix, filename)

def get_data_dirs(base_dir: str) -> List[Tuple[str, str]]:
    data_dirs = os.listdir(base_dir)
    return [
        (os.path.join(base_dir, directory), directory)
        for directory in data_dirs
        if os.path.isfile(
            os.path.join(base_dir, directory, f"{directory}{constants.DENSITY_SUFFIX}.map")
        )
    ]

def get_bounding_box(grid: gemmi.FloatGrid) -> gemmi.PositionBox:
    # Gets box surrounding ASU 
    extent = gemmi.find_asu_brick(grid.spacegroup).get_extent()
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
    return box
 