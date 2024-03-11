import argparse, os
import urllib.request
from tqdm import tqdm 
import multiprocessing

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
                pdb_list.append(pdb.lower().strip("\n"))

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    data_list = [(args.outputdir, pdb) for pdb in pdb_list]

    with multiprocessing.Pool() as pool_:
        x = list(tqdm(pool_.imap_unordered(worker, data_list), total=len(data_list)))

def worker(data):
    output_dir, pdb = data
    output_path = os.path.join(output_dir, f"{pdb}.mtz")
    
    if os.path.exists(output_path):
        return

    try:
        urllib.request.urlretrieve(f"https://edmaps.rcsb.org/coefficients/{pdb}.mtz", 
                                output_path
                                )
    except:
        print("Failed to get pdb - ", pdb)
        return


if __name__ == "__main__":
    main()