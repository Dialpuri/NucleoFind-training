#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gemmi 
import os 
import pandas as pd
from sklearn.model_selection import train_test_split
import multiprocessing
from tqdm import tqdm


# In[3]:


def _initialise_neighbour_search(structure: gemmi.Structure, radius: int = 1.5):
    sugar_neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, radius)
    phosphate_neigbour_search = gemmi.NeighborSearch(
        structure[0], structure.cell, radius
    )
    base_neigbour_search = gemmi.NeighborSearch(structure[0], structure.cell, radius)

    sugar_atoms = ["C1'", "C2'", "C3'", "C4'", "C5'", "O4'"]
    phosphate_atoms = ["P"] # ["O5'", "P", "OP1", "OP2"]
    base_atoms = [
        "C1", "C2", "C3", "C4", "C5","C6", "C7", "C8", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "O2", "O4", "O6"
    ]

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
        sugar_neigbour_search,
        phosphate_neigbour_search,
        base_neigbour_search,
    )


# 
#     # sugar_model = gemmi.Model("A")
#     # phosphate_model = gemmi.Model("A")
#     # base_model = gemmi.Model("A")
# 
#     # sugar_atoms = ["C1'", "C2'", "C3'", "C4'", "C5'", "O4'"]
#     # phosphate_atoms = ["P", "OP1", "OP2"]
#     # base_atoms = [
#     #     "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "O2", "O4", "O6"
#     # ]
# 
#     # for c in st[0]: 
#     #     sug_chain = gemmi.Chain("A")
#     #     pho_chain = gemmi.Chain("A")
#     #     bas_chain = gemmi.Chain("A")
# 
#     #     for r in c: 
#     #         sug_r = r.clone()
#     #         pho_r = r.clone()
#     #         bas_r = r.clone()
# 
#     #         k = gemmi.find_tabulated_residue(r.name)
#     #         for a in r:
#     #             if a.name not in sugar_atoms: 
#     #                 sug_r.remove_atom(a.name, a.altloc, a.element)
#     #             if a.name not in phosphate_atoms: 
#     #                 pho_r.remove_atom(a.name, a.altloc, a.element)
#     #             if a.name not in base_atoms: 
#     #                 bas_r.remove_atom(a.name, a.altloc, a.element)
# 
#     #         sug_chain.add_residue(sug_r)
#     #         pho_chain.add_residue(pho_r)
#     #         bas_chain.add_residue(bas_r)
# 
#     #     sugar_model.add_chain(sug_chain)
#     #     phosphate_model.add_chain(pho_chain)
#     #     base_model.add_chain(bas_chain)
#         
# 
#     # bases = os.path.join(output_dir, "bases.pdb")    
#     # bas_s = gemmi.Structure() 
#     # bas_s.add_model(base_model)
#     # bas_s.cell = grid.unit_cell
#     # bas_s.write_pdb(bases)

# In[4]:


def save_map(array: gemmi.FloatGrid, path: str):
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = array
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(path)


# In[5]:


# data is a tuple of args -> mtz_path: str, pdb_path: str, pdb_code: str, output_dir: str, radius: float = 1.5
def generate_maps(data):
    mtz_path, pdb_path, pdb_code, output_dir, radius = data

    if os.path.exists(output_dir): 
        return

    mtz = gemmi.read_mtz_file(mtz_path)
    st = gemmi.read_structure(pdb_path)

    res = mtz.resolution_high()
    spacing = 0.7
    sample_rate = res/spacing
    
    grid = mtz.transform_f_phi_to_map("FWT", "PHWT", sample_rate=sample_rate)
    grid.normalize()

    sugar_grid = grid.clone()
    sugar_grid.fill(0)
    phosphate_grid = grid.clone()
    phosphate_grid.fill(0)
    base_grid = grid.clone()
    base_grid.fill(0)

    sug, pho, bas = _initialise_neighbour_search(structure=st)
    
    for point in grid:
        position = grid.point_to_position(point)
        any_bases = bas.find_atoms(
            position, "\0", radius=radius
        )
        any_sugars = sug.find_atoms(
            position, "\0", radius=radius
        )
        any_phosphate = pho.find_atoms(
            position, "\0", radius=radius
        )

        base_mask = 1.0 if any_bases else 0.0
        sugar_mask = 1.0 if any_sugars else 0.0
        phosphate_mask = 1.0 if any_phosphate else 0.0

        sugar_grid.set_value(point.u, point.v, point.w, sugar_mask)
        phosphate_grid.set_value(point.u, point.v, point.w, phosphate_mask)
        base_grid.set_value(point.u, point.v, point.w, base_mask)

    os.makedirs(output_dir, exist_ok=True)

    save_map(sugar_grid, os.path.join(output_dir, "sugar.map"))
    save_map(phosphate_grid, os.path.join(output_dir, "phosphate.map"))
    save_map(base_grid, os.path.join(output_dir, "base.map"))
    save_map(grid, os.path.join(output_dir, "experimental.map"))


# In[6]:


def captured_worker(data):
    try: 
        generate_maps(data)
    except Exception as e:
        print(e)
        return


# In[ ]:


# type = "na_only"
# df = pd.read_csv(f"/vault/NucleoFind-training-pr/data/{type}/refined_paths.csv")
# train, test = train_test_split(df, test_size=0.2)
# radius = 1.5
# train_data = [(mtz_path, pdb_path, pdb, f"/vault/NucleoFind-training-pr/train_maps/{type}/{pdb}", radius) for index, (pdb, mtz_path, pdb_path) in train.iterrows()]
# test_data = [(mtz_path, pdb_path, pdb,f"/vault/NucleoFind-training-pr/test_maps/{type}/{pdb}", radius) for index, (pdb, mtz_path, pdb_path) in test.iterrows()]


# with multiprocessing.Pool(8) as p:
#     x = list(tqdm(
#         p.imap_unordered(captured_worker, train_data), total=len(train_data)
#     ))

# with multiprocessing.Pool(8) as p:
#     x = list(tqdm(
#         p.imap_unordered(captured_worker, test_data), total=len(test_data)
#     ))


# In[ ]:


# type = "protein_na"
# df = pd.read_csv(f"/vault/NucleoFind-training-pr/data/{type}/refined_paths.csv")
# train, test = train_test_split(df, test_size=0.2)

# radius = 1.5
# ref_train_data = [(mtz_path, pdb_path, pdb, f"/vault/NucleoFind-training-pr/train_maps/{type}/{pdb}", radius) for index, (pdb, mtz_path, pdb_path) in train.iterrows()]
# ref_test_data = [(mtz_path, pdb_path, pdb,f"/vault/NucleoFind-training-pr/test_maps/{type}/{pdb}", radius) for index, (pdb, mtz_path, pdb_path) in test.iterrows()]

# with multiprocessing.Pool(8) as p:
#     x = list(tqdm(
#         p.imap_unordered(captured_worker, ref_train_data), total=len(ref_train_data)
#     ))

# with multiprocessing.Pool(8) as p:
#     x = list(tqdm(
#         p.imap_unordered(captured_worker, ref_test_data), total=len(ref_test_data)
#     ))


# In[12]:
# type = "protein_na"
# df = pd.read_csv(f"/vault/NucleoFind-training-pr/data/{type}/refined_paths.csv")

# test = []
# a,b = 0,0
# radius = 1.5
# for index, i in df.iterrows(): 
#     pdb = i["PDB"]
#     mtz_path = i["MTZPath"]
#     pdb_path = i["PDBPath"]
#     path = f"/vault/NucleoFind-training-pr/train_maps/protein_na/{pdb}"
#     if not os.path.exists(path):
#        test.append((mtz_path, pdb_path, pdb,f"/vault/NucleoFind-training-pr/test_maps/{type}/{pdb}", radius))

# with multiprocessing.Pool(8) as p:
#      x = list(tqdm(
#          p.imap_unordered(captured_worker, test), total=len(test)
#      ))


df = pd.read_csv(f"/vault/NucleoFind-training-pr/data/protein_na/refined_paths.csv")
radius = 1.5

type="protein_na_unref_none"
# unref_train_data = []

# for path in os.scandir("/vault/NucleoFind-training-pr/train_maps/protein_na"):
#     pdb = path.name
#     row = df[df["PDB"]==pdb]
#     unrefined_base_path = "/vault/NucleoFind-training-pr/unrefined_data_b"
#     data_dir = os.path.join(unrefined_base_path, pdb, "0", f"{pdb}_0.mtz")
#     if os.path.exists(data_dir):
#         unref_train_data.append((data_dir, row["PDBPath"].values[0], pdb,  f"/vault/NucleoFind-training-pr/train_maps/{type}/{pdb}", radius))

unref_test_data = []
for path in os.scandir("/vault/NucleoFind-training-pr/test_maps/protein_na"):
    pdb = path.name
    row = df[df["PDB"]==pdb]
    unrefined_base_path = "/vault/NucleoFind-training-pr/unrefined_data_b"
    data_dir = os.path.join(unrefined_base_path, pdb, "0", f"{pdb}_0.mtz")
    if os.path.exists(data_dir):
        unref_test_data.append((data_dir, row["PDBPath"].values[0], pdb,  f"/vault/NucleoFind-training-pr/test_maps/{type}/{pdb}", radius))

# print(unref_test_data)

# with multiprocessing.Pool(8) as p:
#     x = list(tqdm(
#         p.imap_unordered(captured_worker, unref_train_data), total=len(unref_train_data)
#     ))

with multiprocessing.Pool(12) as p:
    x = list(tqdm(
        p.imap_unordered(captured_worker, unref_test_data), total=len(unref_test_data)
    ))

