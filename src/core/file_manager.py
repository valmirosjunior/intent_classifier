from os import listdir
from os.path import isfile, join
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

__PROJECT_DIR = str(Path(__file__).parent.parent.parent)


def get_project_dir():
    return __PROJECT_DIR


def filename_from_project_dir(filename):
    return f'{get_project_dir()}/{filename}'


def filename_from_data_dir(filename):
    return filename_from_project_dir(f'data/{filename}')


def read_dataframes_of_dir(dir_path):
    files = listdir(dir_path)

    filenames = [dir_path + file for file in files if isfile(join(dir_path, file))]

    filenames_ordered = sorted(filenames)

    data_frames = [pd.read_json(file) for file in filenames_ordered]

    return pd.concat(data_frames)


def read_multiple_files(filenames: list[str]) -> pd.DataFrame:
    return pd.concat(Parallel(n_jobs=-1)(delayed(read_json_file)(file_path) for file_path in filenames),
                     ignore_index=True)


def read_json_file(path):
    return pd.read_json(path, orient='records', dtype={
        'txt': str,
        'embeddings': np.array
    }, lines=True)
