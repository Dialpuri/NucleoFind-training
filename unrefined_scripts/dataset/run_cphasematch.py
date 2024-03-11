import subprocess
from typing import Dict
def run(mtz_in: str) -> Dict[str,str]: 
    # print("Running CPHASEMATCH with", mtz_in)
    _args_ = ["-mtzin", mtz_in, "-colin-fo", "FP,SIGFP", "-colin-fc-1" ,"FWT,PHWT" ,"-colin-fc-2" ,"FWT2,PHWT2"]
    p = subprocess.Popen(
         args=["/opt/xtal/ccp4-8.0/bin/cphasematch"] + _args_,
                    #  shell=True, 
                     stdout=subprocess.PIPE)
    out, err = p.communicate()
    out_ = out.decode()

    split = out_.split("\n")
    read_further = False

    data = []

    for x in split:
        if 'Overall statistics:' in x:
            read_further=True
            continue

        if read_further:
            if '<B>' in x:
                read_further = False
                continue
        
            data.append(x)

    if len(data) != 3: 
        print("Something has gone wrong!, data is ", data)
        return
    
    data = [x.lstrip(" ") for x in data[:-1]]
    data = [x.split(" ") for x in data]
    data = [[y for y in x if y] for x in data]
    data = dict(zip(*data))
    return data
       
if __name__ == "__main__":
    run(mtz_in="phase_dependence_experiment/combined_2.mtz")