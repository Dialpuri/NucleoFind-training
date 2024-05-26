import remove_na_and_water as remove_atoms
import run_refmac as refmac
import run_cad as cad
import run_cphasematch as cphasematch
import os
import json
import pandas as pd
from tqdm import tqdm
import multiprocessing


# def main(): 
#     df = pd.read_csv("data/raw/protein_na_pdblist.csv")
#     pdbs = df["PDB ID"].to_list()
#     data = [("data/protein_na", pdb.lower()) for pdb in pdbs]
    
#     with multiprocessing.Pool(12) as p:
#         x = list(tqdm(p.imap_unordered(worker, data), total=len(data)))


def main(): 
    df = pd.read_csv("data/raw/reserved_examples/reserved_protein_na_pdblist.csv")
    pdbs = df["PDB ID"].to_list()
    data = [("data/protein_na_reserved", pdb.lower()) for pdb in pdbs]
    
    with multiprocessing.Pool(12) as p:
        x = list(tqdm(p.imap_unordered(worker, data), total=len(data)))

def worker(data):

    in_dir, pdb = data

    cutoffs=[0]

    pdb_in=f"{in_dir}/pdb/{pdb}.pdb"
    mtz_in=f"{in_dir}/mtz/{pdb}.mtz"

    if not os.path.exists(pdb_in) or not os.path.exists(mtz_in):
        return 

    data_dir = "unrefined_reserved_data"
    fmap_scores=[]

    stats = {"pdb": pdb, "cutoffs": []}
    stats_dir = os.path.join(data_dir, pdb, "stats.json")

    for cutoff in cutoffs:
        cutoff_dir = os.path.join(data_dir, pdb, str(cutoff))
        
        os.makedirs(cutoff_dir, exist_ok=True)

        cutoff_pdb_out = os.path.join(cutoff_dir, f"{pdb}_{cutoff}.pdb")

        refmac_name_out = os.path.join(cutoff_dir)
        refmac_mtz_out = os.path.join(refmac_name_out, f"{pdb}_{cutoff}.mtz")
        combined_mtz_out = f"{refmac_name_out}_combined.mtz"

        remove_atoms.run(pdb_in=pdb_in, pdb_out=cutoff_pdb_out, cutoff=cutoff)
        refmac.run(mtz_in=mtz_in, pdb_in=cutoff_pdb_out, name=f"{pdb}_{cutoff}", mtz_path_out=cutoff_dir, other_path_out="unrefined_reserved_data/bin")

        if not os.path.exists(refmac_mtz_out):
            return

        cad.run(mtz1_in=mtz_in, mtz2_in=refmac_mtz_out, mtz_out=combined_mtz_out)
        data = cphasematch.run(combined_mtz_out)
        stats["cutoffs"].append({"cutoff": cutoff, "data": data})

        try:
            files = os.scandir("/vault/NucleoFind-training-pr/unrefined_reserved_data/bin")
            for f in files:
                if os.path.exists(f):
                    os.remove(f)
        except:
            continue

    with open(stats_dir, "w") as j:
        json.dump(stats, j, indent=4)

if __name__ == "__main__":
    # worker(("data/protein_na", "1a02"))
    main()