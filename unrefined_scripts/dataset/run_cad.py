import subprocess
import os 

def run(mtz1_in: str, mtz2_in: str, mtz_out: str): 
    # print("Running CAD with", mtz1_in, "and", mtz2_in)
    _args = []
    _args += ["hklin1", mtz1_in]
    _args += ["hklin2", mtz2_in]
    
    _args += ["HKLOUT", mtz_out]
   
    _stdin = []
    _stdin.append("LABIN FILE 1 E1=FP E2=SIGFP E3=FWT E4=PHWT")
    _stdin.append("LABIN FILE 2 E1=FWT E2=PHWT")
    _stdin.append("LABOUT FILE 2 E1=FWT2 E2=PHWT2")
    _stdin.append("END")

    process = subprocess.Popen(
    args=["/opt/xtal/ccp4-8.0/bin/cad"] + _args,
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


if __name__ == "__main__":
    run("","","")

