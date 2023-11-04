import pandas as pd
import constants
import os
from sklearn.model_selection import train_test_split


def combine_help_files():
    base_dir = constants.DATASET_DIRECTORY

    main_df = pd.DataFrame(columns=["PDB", "X", "Y", "Z"])

    for dir in os.scandir(base_dir):
        context_path = os.path.join(dir.path, "seeded_feature_list.csv")
        df = pd.read_csv(context_path)
        df = df.assign(PDB=dir.name)
        main_df = pd.concat([main_df, df])

    main_df.to_csv(
        f"{constants.TRAINING_DATA_OUT}/combined_feature_list.csv", index=False
    )


def generate_test_train_split():
    df = pd.read_csv(f"{constants.TRAINING_DATA_OUT}/combined_feature_list.csv")

    train, test = train_test_split(df, test_size=0.2)

    train.to_csv(f"{constants.TRAINING_DATA_OUT}/train.csv", index=False)
    test.to_csv(f"{constants.TRAINING_DATA_OUT}/test.csv", index=False)
