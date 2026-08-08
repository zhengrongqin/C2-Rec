

Esta es una implementación de TensorFlow para C2-Rec

Hacemos referencia al repositorio [SASRec](https://github.com/kang205/SASRec).

## Requisitos

requirement.txt

## Conjuntos de datos

Este repositorio incluye el conjunto de datos Amazon Beauty como ejemplo. También puedes descargar los datos de reseñas de Amazon desde *[aquí](http://jmcauley.ucsd.edu/data/amazon/index.html)*.

## Entrenamiento del Modelo

Para entrenar el modelo en `Beauty`:

```
python main.py \
--dataset=Beauty \
--train_dir=model_train \
--maxlen=50 \
--dropout_rate=0.5 \
--con_alpha=5.0 \
--rd_alpha=1.0 \
--neg_test=500 \
--user_reg_type=kl \
--lr=0.001 \
--rd_reduce=mean \
--neg_sample_n=50
```
