import subprocess
import os 

def run(mtz_in: str, pdb_in: str, name: str, mtz_path_out: str, other_path_out: str): 
    # print("Running REFMAC with", mtz_in, pdb_in)
    _args = []
    _args += ["HKLIN", mtz_in]
    _args += ["XYZIN", pdb_in]
    
    _args += ["HKLOUT", f"{mtz_path_out}/{name}.mtz"]
    _args += ["XYZOUT", f"{other_path_out}/{name}.cif"]
    _args += ["XMLOUT", f"{other_path_out}/{name}.xml"]
    labin = "FP=FP"
    labin += " SIGFP=SIGFP"
    labin += " FREE=FREE"
    _stdin = []
    _stdin.append("LABIN " + labin)
    _stdin.append(f"NCYCLES 1")
    _stdin.append("WEIGHT AUTO")
    _stdin.append("MAKE HYDR NO")
    _stdin.append("MAKE NEWLIGAND NOEXIT")
    _stdin.append("PHOUT")
    _stdin.append("PNAME modelcraft")
    _stdin.append("DNAME modelcraft")
    _stdin.append("END")

    process = subprocess.Popen(
    args=["/opt/xtal/ccp4-8.0/bin/refmac5"] + _args,
    stdin=subprocess.PIPE if _stdin else None,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    encoding="utf8",
    env={**os.environ,},
    cwd=os.getcwd(),
    )
    if _stdin:
        stdin_str = '\n'.join(_stdin)
        process.communicate(input=stdin_str)

