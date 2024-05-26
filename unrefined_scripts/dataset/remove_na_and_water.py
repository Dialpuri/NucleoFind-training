import gemmi
import math
import numpy as np

def run(pdb_in: str, pdb_out: str, cutoff: float):
    # print("Running remove_na with", pdb_in, "cutoff=", cutoff)
    try:
        s = gemmi.read_structure(pdb_in)
    except:
        print("Failed to read", pdb_in)
        return False

    ns = gemmi.Structure()
    ns.cell = s.cell
    nm = gemmi.Model(s[0].name)

    to_remove_count = 0
    for c in s[0]:
        for r in c: 
            i = gemmi.find_tabulated_residue(r.name)
            if i.is_nucleic_acid(): 
                to_remove_count+=1
    
    threshold = math.floor(cutoff * to_remove_count)

    count = 0

    residue_bfactors = []

    for c in s[0]:
        nc = gemmi.Chain(c.name)
        for r in c: 
            i = gemmi.find_tabulated_residue(r.name)
            if i.is_water(): 
                continue

            if i.is_nucleic_acid(): 
                count +=1
                if count > threshold:
                    continue
            
            residue_avg = np.average([a.b_iso for a in r])
            residue_bfactors.append(residue_avg)

    avg_b = np.average(residue_bfactors)

    for c in s[0]:
        nc = gemmi.Chain(c.name)
        for r in c: 
            i = gemmi.find_tabulated_residue(r.name)
            if i.is_water(): 
                continue

            if i.is_nucleic_acid(): 
                count +=1
                if count > threshold:
                    continue
            
            for a in r: 
                a.b_iso = avg_b

            nc.add_residue(r)
        nm.add_chain(nc)
    ns.add_model(nm)
    ns.write_pdb(pdb_out)

    return True
    