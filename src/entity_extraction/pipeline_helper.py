import json

import pandas as pd

from . import extract_entity_helper as eh
from . import ignorer_helper as ih


def mark_entities(text, entities):
    clean_entities = ih.remove_duplicated_entities(json.loads(entities))

    entities_sorted = sorted(clean_entities, key=lambda x: x['start'], reverse=True)

    for ent in entities_sorted:
        start_ent = ent['start']
        end_ent = ent['end']
        label = ent['entity']

        text = text[:start_ent] + '[' + text[start_ent:end_ent] + '](' + label + ')' + text[end_ent:]

    return text


def run_pipeline(path_input, path_output):
    print('Reading file...')
    df = pd.read_csv(path_input)

    print('Adding entities...')
    df['ents'] = df.apply(lambda r: eh.extract_entities(r['txt'], serialize=True), axis=1)

    print('Formatting entities...')
    df['ents_old'] = df['ents']
    df['ents'] = df['ents'].apply(ih.format_entities)

    print('Annotating entities...')
    df['annotated_txt'] = df.apply(lambda r: mark_entities(r['txt'], r['ents']), axis=1)

    df[['txt', 'annotated_txt']].to_csv(path_output, index=False)

    return df
