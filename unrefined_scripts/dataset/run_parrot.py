
import subprocess
import os 
import modelcraft as mc
import gemmi 

def run(mtz_in: str, pdb_in: str, contents_path: str, hklout_name: str): 
    # print("Running parrot with", mtz_in)


    asu = mc.AsuContents.from_file(contents_path)
    mtz = gemmi.read_mtz_file(mtz_in)
    solvent_content = mc.solvent.solvent_fraction(asu, mtz)

    _args = []
    _args += ["-mtzin", mtz_in]
    _args += ["-colin-fo", "FP,SIGFP"]
    _args += ["-colin-free", "FREE"]
    phases_arg = "-colin-hl"
    phases_label = "HLACOMB,HLBCOMB,HLCCOMB,HLDCOMB"
    _args += [phases_arg, phases_label]
    _args += ["-colin-fc", "FC_ALL_LS,PHIC_ALL_LS"]
    _args += ["-pdbin-mr", pdb_in]
    _args += ["-solvent-content", "%.3f" % solvent_content]
    _args += ["-cycles", "5"]
    _args += ["-anisotropy-correction"]
    _args += ["-mtzout", f"{hklout_name}.mtz"]

    process = subprocess.Popen(
    args=["/jarvis/programs/xtal/ccp4-8.0/bin/cparrot"] + _args,
    stdin=None,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    encoding="utf8",
    env={**os.environ,},
    cwd=os.getcwd(),
    )


if __name__ == "__main__":
    run(
        mtz_in = "data/unphased_mtz/1a02.mtz",
        pdb_in = "/old_vault/pdb/pdb1a02.ent",
        contents_path = "data/modelcraft_contents/1a02.json",
        hklout_name = "data/postparrotmtz/1a02"
    )