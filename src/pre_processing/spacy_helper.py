import json

import spacy

__nlp = spacy.load("pt_core_news_lg")


def is_loc_or_person(text):
    doc = __nlp(text)
    entities = ["LOC", "PER"]
    result = ''.join([
        (
            "PER" if token.ent_type_ == "PER" else
            "LOC" if token.ent_type_ == "LOC" else
            token.text
        ) + token.whitespace_
        for index, token in enumerate(doc) if
        index == 0 or token.ent_type_ not in entities or doc[index - 1].ent_type_ != token.ent_type_
    ])

    return result.strip() in entities


def extract_ents(text):
    ents = [[ent.label_, ent.text, ent.start_char, ent.end_char] for ent in nlp(text).ents]

    data = {}
    for ent in ents:
        label = ent[0]
        if label not in data:
            data[label] = []
        data[label].append([ent[1], ent[2], ent[3]])

    return json.dumps({'spacy': data}, ensure_ascii=False)


def replace_entities(text):
    doc = __nlp(text)

    result = ''.join([
        (
            "PER" if token.ent_type_ == "PER" else
            "LOC" if token.ent_type_ == "LOC" else
            token.text
        ) + token.whitespace_
        for index, token in enumerate(doc) if
        index == 0 or token.ent_type_ not in ["PER", "LOC"] or doc[index - 1].ent_type_ != token.ent_type_
    ])

    return result.strip()
