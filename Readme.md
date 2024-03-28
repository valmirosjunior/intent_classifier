## About the Project
This project was developed within the scope of the Master's degree in Computer Science at [Universidade Federal do Ceará de Quixadá](https://www.quixada.ufc.br)

## Important notes
- Pay attention in torch installation for your machine,
maybe it'll be necessary to check the torch website for instructions
For my machine it was this command:
- pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113


### How to generate nlu models

- Just run this after generate the nlu data, and split it\
using the [nlu_builder.ipynb](notebooks/nlu_builder.ipynb).

```shell
time ./build_nlu.sh `pwd`/data/nlu_models/patient
time ./build_nlu.sh `pwd`/data/nlu_models/patient/without_others_intent
```

### How to valid data of an model nlu

```shell
rasa test nlu --nlu data/output/patient/without_others_intent/k100_without_sentences_higher_than_median/intersection_300_sentences_with_label.yml -m data/output/patient/without_others_intent/k100_without_sentences_higher_than_median/flair_pt/rasa/nlu-20220909-011126.tar.gz
```