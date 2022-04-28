import json

import spacy

from src.core import file_manager as fm

nlp1 = spacy.load(fm.filename_from_project_dir("src/entity_extraction/pt_models_api_boletim/ner_output5"))
nlp2 = spacy.load(fm.filename_from_project_dir("src/entity_extraction/pt_models_api_boletim/ner_output9"))


def extract_entities_by_model(model, text):
    entities = [[ent.label_, ent.text, ent.start_char, ent.end_char] for ent in model(text).ents]

    data = {}
    for ent in entities:
        label = ent[0]
        if label not in data:
            data[label] = []
        data[label].append([ent[1], ent[2], ent[3]])

    return data


def extract_entities(txt, serialize=True):
    text = str(txt)

    entities = {
        'm1': extract_entities_by_model(nlp1, text),
        'm2': extract_entities_by_model(nlp2, text),
    }

    return json.dumps(entities, ensure_ascii=False) if serialize else entities
