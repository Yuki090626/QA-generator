# QA-generator

## Model
Info-HCVAE [[paper]](https://www.aclweb.org/anthology/2020.acl-main.20/)

https://github.com/seanie12/Info-HCVAE


## Dependencies
- python >= 3.6
- pytorch >= 1.4
- [pytorch_scatter](https://github.com/rusty1s/pytorch_scatter)


## Setup
~~~
cd QA_gen
pip install -r requirements.txt
~~~
 

## Run
pre-trainedモデル(best_f1_model.pt)を[ここ](https://drive.google.com/file/d/1UfkMBDrK-oI0m-HCe0VSICjbEHDdI_Y9/view?usp=sharing)からダウンロードして、```QA_gen/media/model/```に配置してください
~~~
cd QA_gen
python manage.py migrate
python manage.py runserver
~~~


## Notebooks
Directory : ```model/vae/```

- create_document.ipynb

  Documentの作成

- train-japanese.ipynb

  Info-HCVAEモデルの学習

- eval-japanese.ipynb 

  Info-HCVAEモデルの評価・QA生成
