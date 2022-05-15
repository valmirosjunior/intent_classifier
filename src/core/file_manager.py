from glob import glob
from pathlib import Path

import pandas as pd

__PROJECT_DIR = str(Path(__file__).parent.parent.parent)


def get_project_dir():
    return __PROJECT_DIR


def filename_from_project_dir(filename):
    return f'{get_project_dir()}/{filename}'


def filename_from_data_dir(filename):
    return filename_from_project_dir(f'data/{filename}')


def read_json_of_dir(dir_path):
    filenames = glob(f'{dir_path}/*.json')

    filenames_ordered = sorted(filenames)

    data_frames = [pd.read_json(file) for file in filenames_ordered]

    return pd.concat(data_frames)
