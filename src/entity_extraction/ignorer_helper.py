import numpy as np
import pandas as pd
import json
import re
from collections import OrderedDict
import matplotlib.pyplot as plt
import spacy


incorrect_classification_m1 = [
 "Tudo", "hapvida", "pedra", "sontoma", "dnv", "harvonio", "ce", "põe", "som",
 "francisco", "miriã", "miosan para dores", "cha de hortelã", "dificuldade", "orientaçã",
 "pensamento", "nariz estar", "sociológico", "radiologia", "aaah", "entonação", "baby",
 "boa-noite", "trampo", "distanciar", "ite", "gota", "elogia", "rebite", "pertubam",
 "hipnose", "perdoe", "temperarura", "masmo", "falta", "descrição", "sem para",
 "charope", "acelerace", "polivitaminico", "aglomeraçães", "no", "mayra", "obrigafo",
 "avisando", "mortes", "alcimar", "26|04|1995", "transmetindo", "parassetamol",
 "para", "aperto", "herminio", "lígia", "parecetamol", "canindezinho", "pici",
 "parangaba", "lasortana", "antiflamatorios", "pastoso", "tolrest", "gonzaguinha",
 "unha", "paulo", "naiara", "27/04", "clenil", "tbm", "morte", "peso", "queixa",
 "hidrata", "começaria", "conversarmos", "aglomeração", "aproveite", "obviamente",
 "estratégia", "demonstre", "esclarecendo", "pray"
]

incorrect_classification_m2 = [
 "maresis", "doença para caso", "sontoma", "dnv", "som", "tudo", "damas", "boca",
 "luzia", "rose", "sem", "lígia", "evandyra", "não", "tolrest", "trampo", "xau",
 "distanciar", "sinusitemuito", "ainda", "ite", "elogia", "lembrada", "hipnose",
 "perdoe", "nça", "masmo", "falta", "terra", "miriã", "descrição", "diq", "quintira",
 "resperia", "lágrimas", "noitefabiana", "gorduroso", "polivitaminico", "cefaliv",
 "t.", "mortes", "alcimar", "pastoso", "fonseca", "evacuações", "zelia correia",
 "dificultasse", "neorosoro", "histerectomia", "eliana", "charope", "visão", "hiato",
 "desesperança", "paralisa", "perda de  ", "ok", "pedra", "versicula", "em", "ce",
 "canindezinho", "de", "bolhas", "lasortana", "aglomeração", "gonzaguinha", "antiflamatorios",
 "paulo", "parangaba", "unha", "parecetamol", "queixa", "naiara", "tbm", "clenil",
 "morte", "peso","gorduroso","paulo","antônia","telefônica","queixa","eglivannia",
 "parangaba", "síndrome de   "
]

ignored_entities = set(incorrect_classification_m1+incorrect_classification_m2)

ignored_entities = list(map(lambda x: x.lower(), ignored_entities))


def remove_duplicated_entities(entities):
  indexes_to_remove = []

  for i in range(len(entities) -1):
    for j in range(i + 1, len( entities)):
      if entities[i]['start'] == entities[j]['start']:
        index_to_remove = i if entities[i]['end'] < entities[j]['end'] else j
        indexes_to_remove.append(index_to_remove)
      elif entities[i]['end'] == entities[j]['end']:
        index_to_remove = i if entities[i]['start'] > entities[j]['start'] else j
        indexes_to_remove.append(index_to_remove)

  return [ent for index, ent in enumerate(entities) if index not in indexes_to_remove]

def remove_ignored_entities(entities):
  for model in entities.keys():
    for label in entities[model].keys():
      filtered_entities = []
      for index, entity in enumerate(entities[model][label]):
        if entity[0].lower() not in ignored_entities:
          filtered_entities.append(entity)

      entities[model][label] = filtered_entities

  return entities

def merge_entities(json_entities):
  entities = []
  for model in ['m1', 'm2']:
  # for model in json_entities.keys():
    for label in json_entities[model].keys():
      for entity in json_entities[model][label]:
        entities.append({
            'entity': label,
            'value': entity[0],
            'start': entity[1],
            'end': entity[2],
        })

  return entities

def format_entities(json_entities):
  entities = json.loads(json_entities)
  entities = remove_ignored_entities(entities)
  entities = merge_entities(entities)
  entities = remove_duplicated_entities(entities)

  return json.dumps(entities, ensure_ascii=False)