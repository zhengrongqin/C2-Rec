This is a TensorFlow implementation for C2-Rec

We refer to the repo [SASRec](https://github.com/kang205/SASRec).

## Requirements

requirement.txt

## Datasets

This repo includes Amazon Beauty dataset as an example. you could also download Amazon review data from *[here](http://jmcauley.ucsd.edu/data/amazon/index.html)*.

## Model Training

To train model on `Beauty`: 

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

