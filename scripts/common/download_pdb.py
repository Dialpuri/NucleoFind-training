import argparse, os
import urllib.request
from tqdm import tqdm 

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("-pdblist",help="file containg a list of comma seperated PDB IDs")
    parser.add_argument("-outputdir", help="output directory")

    args = parser.parse_args()

    if not os.path.exists(args.pdblist): 
        raise RuntimeError("Could not find PDB list path")
    
    with open(args.pdblist, encoding='UTF-8') as pdb_file: 
        data = pdb_file.readlines()
        pdb_list = []
        for line in data:
            for pdb in line.split(","):
                pdb_list.append(pdb.lower())

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    for pdb in tqdm(pdb_list): 
            
        output_path = os.path.join(args.outputdir, f"{pdb}.mtz")
        
        if os.path.exists(output_path):
            continue

        try:
            urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb.upper()}.pdb", 
                                   output_path
                                   )
        except:
            continue


if __name__ == "__main__":
    main()