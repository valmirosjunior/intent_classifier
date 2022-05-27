## Important notes
- Pay attention in torch installation for your machine,
maybe it'll be necessary to check the torch website for instructions


- How to generate nlu models just run this after generate the nlu data, and split it\
using the [nlu_builder.ipynb](notebooks/nlu_builder.ipynb).
```shell
./build_nlu.sh `pwd`/data/nlu_models/patient/k100/bert_pt
```

- With the update:
```shell
time ./build_nlu.sh `pwd`/data/nlu_models/patient/without_others_intent
```

