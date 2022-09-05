import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.core import file_manager as fm
from src.embeddings.constants import EMBEDDING_MODELS_TRANSLATION


def add_suffix_name(filename, suffix):
    segments = filename.split('/')

    return f"{'/'.join(segments[:-1])}/{suffix}{segments[-1]}"


def split_data(path_data):
    df = pd.read_csv(path_data)

    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

    return df_train, df_test


def fix_punctuation_after_dot(example):
    fixed_example = example

    matches = [x.span() for x in re.finditer(r'\.[A-z]', example)]

    for m in reversed(matches):
        fixed_example = fixed_example[:m[0] + 1] + ' ' + fixed_example[m[1] - 1:]

    return fixed_example


def getting_intents(df):
    grouped = df.groupby(['intent'])

    return grouped['annotated_txt'].apply(list).to_dict()


def convert_to_nlu_format(df):
    print('getting intents...')

    data = getting_intents(df)

    lines = [
        'version: \'2.0\'',
        '',
        'nlu:'
    ]

    for intent in data.keys():
        lines.append(f'- intent: {intent}')
        lines.append('  examples: |')

        for example in data[intent]:
            fixed_example = fix_punctuation_after_dot(example)

            lines.append(f'    - {fixed_example}')

        print(f'The intent: {intent}, has {len(data[intent])} examples')

    return lines


def generate_nlu_file_from_df(df, path_output_nlu):
    print(f'generating {path_output_nlu}')

    lines = convert_to_nlu_format(df)

    formatted_data = "\n".join(lines)

    f = open(path_output_nlu, "w+")
    f.write(formatted_data)
    f.close()
    print('the content was saved in:', path_output_nlu)


def run_pipeline(actor, subfolder):
    for model in EMBEDDING_MODELS_TRANSLATION.keys():
        path = fm.filename_from_data_dir(f'output/{actor}/{subfolder}/{model}/annotated_sentences.csv')

        df_train, df_test = split_data(path)

        output_dir = Path(fm.filename_from_data_dir(f'nlu_models/{actor}/{subfolder}/{model}'))
        output_dir.mkdir(parents=True, exist_ok=True)

        generate_nlu_file_from_df(df_train, f'{output_dir}/training_data.yml')
        print('======================================================')

        generate_nlu_file_from_df(df_test, f'{output_dir}/test_data.yml')
        print('\n')


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Process some integers.')
#     parser.add_argument('-a', '--actor', help='The actor to generate de nlu.yml data', required=True)
#     parser.add_argument('-s', '--subfolder', help='The subfolder to generate de nlu.yml data', required=True)
#
#     args = parser.parse_args()
#
#     run_pipeline(actor=args.actor, subfolder=args.subfolder)


