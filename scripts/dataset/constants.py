
DENSITY_SUFFIX = "_density"
SUGAR_SUFFIX = "_sugar"
PHOSPHATE_SUFFIX = "_phosphate"
BASE_SUFFIX = "_base"
DATA_PATH = "data"
DATASET_DIRECTORY = "training_dataset"
TRAINING_DATA_OUT = "training_data"


def get_sugar_atoms(): 
    return ["C1'", "C2'", "C3'", "C4'", "C5'", "O4'"]

def get_phosphate_atoms(): 
    return ["P", "OP1", "OP2", "O5'", "O3'"]

def get_base_atoms(): 
    return [
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "N1",
        "N2",
        "N3",
        "N4",
        "N5",
        "N6",
        "N7",
        "N8",
        "N9",
        "O2",
        "O4",
        "O6",
    ]

