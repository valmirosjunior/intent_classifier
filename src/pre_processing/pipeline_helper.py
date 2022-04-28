import numpy as np
import pandas as pd

from src.core import file_manager as fm
from src.pre_processing import filters_helper as fh
from src.pre_processing import regex_helper as rh
from src.pre_processing import spacy_helper as sh


def __find_better_conversations(df):
    df['created_at'] = pd.to_datetime(df['cri']).dt.strftime("%y-%m-%dT%H:%M")

    grouped_conversations = df.drop_duplicates(
        subset=['idp', 'created_at', 'txt'], keep="last"
    ).sort_values(['idp', 'cri']).groupby(by="idp")

    better_conversations = grouped_conversations.filter(lambda x: fh.is_maximum_note(x) or fh.is_questions_answered(x))

    better_conversations_idps = better_conversations.idp.unique()

    conversations = df[df['idp'].isin(better_conversations_idps)]

    return conversations


def __get_human_interactions(conversations):
    conversations[['id', 'seq']] = conversations['_id'].str.split("-", expand=True)
    conversations['seq'] = conversations['seq'].astype('int64')
    conversations = conversations.groupby(by=['id'])

    conversations = conversations.filter(lambda x: set(x['ori']) == set(['patient', 'human']))

    return conversations


def __pre_process_conversations(conversations_df):
    print('Saving original txt')
    conversations_df['original_txt'] = conversations_df['txt']

    print('Removing emojis....')
    conversations_df['txt'] = conversations_df['txt'].apply(rh.remove_emoji)

    print('Removing break lines....')
    conversations_df['txt'] = conversations_df['txt'].apply(rh.remove_break_lines)

    print('Dropping duplicates....')
    conversations_df = conversations_df.drop_duplicates(subset=['txt'])

    print('Filtering number values....')
    conversations_df = conversations_df[conversations_df.apply(lambda x: not rh.is_number(x['txt']), axis=1)]

    print('Filtering link values....')
    conversations_df = conversations_df[conversations_df.apply(lambda x: not rh.is_link(x['txt']), axis=1)]

    print('Filtering loc or per values....')
    conversations_df = conversations_df[conversations_df.apply(lambda x: not sh.is_loc_or_person(x['txt']), axis=1)]

    print('replacing empty values...')
    conversations_df = conversations_df.replace('', np.nan)

    print('Dropping null values....')
    conversations_df = conversations_df.dropna()

    return conversations_df


def __store_data(conversations):
    conversations.to_csv(fm.filename_from_data_dir('output/clean_conversations.csv'), index=False)

    conversations[conversations['ori'] == 'patient'].to_csv(
        fm.filename_from_data_dir('output/patient/clean_sentences.csv'), index=False
    )

    conversations[conversations['ori'] == 'human'].to_csv(
        fm.filename_from_data_dir('output/professional/clean_sentences.csv'), index=False
    )


def __describe_dataset(dataset):
    print('\n=========================================')
    print(f"Total de sentenças: {dataset['txt'].count()}")
    print(f"Total de sentenças dos pacientes: {dataset[dataset['ori'] == 'patient']['txt'].count()}")
    print(f"Total de sentenças dos atendentes: {dataset[dataset['ori'] == 'human']['txt'].count()}")
    print(f"Total de diálogos: {len(dataset['idp'].unique())}")    


def run_pipeline(input_dir):
    print('reading the data...')
    df = fm.read_dataframes_of_dir(input_dir)

    print('Getting the better conversations...')
    better_conversations = __find_better_conversations(df)

    print('Getting human interactions...')
    conversations = __get_human_interactions(better_conversations)

    sorted_conversations = conversations.sort_values(['idp', 'cri'])

    print('Pre processing conversations...')
    conversations = __pre_process_conversations(sorted_conversations)

    print('Storing data...')
    __store_data(conversations)

    print('Describing data...')
    __describe_dataset(df)

    __describe_dataset(conversations)

    print('The pipeline has been fineshed!')
    return df, conversations
