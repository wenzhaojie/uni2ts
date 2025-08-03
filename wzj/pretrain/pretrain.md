# 示例

```python
python -m cli.train \
  -cp conf/pretrain \
  run_name=moirai_small \
  model=moirai_small \
  data=lotsa_v1_weighted \
  trainer.max_epochs=1000 \
  train_dataloader.batch_size=256

```
```python
python -m cli.train \
  -cp conf/pretrain \
  run_name=moirai_base \
  model=moirai_base \
  data=lotsa_v1_weighted \
  trainer.max_epochs=10_000 \
  train_dataloader.batch_size=128
```


# PackedValueOrientedMAELoss
```python
python -m cli.train \
  -cp conf/pretrain \
  run_name=moirai_small \
  model=moirai_small_wzj_value_loss \
  data=lotsa_v1_weighted \
  trainer.max_epochs=1000 \
  train_dataloader.batch_size=256
```

# eval
```python
python -m cli.eval \
  run_name=example_eval_moirai_small_wzj_value_loss \
  model=moirai_small_wzj_value_loss \
  model.patch_size=32 \
  model.context_length=1000 \
  data=etth1_test
```

# PackedValueOrientedMAELoss 训练一个base的模型
```python
python -m cli.train \
  -cp conf/pretrain \
  run_name=moirai_base \
  model=moirai_base_wzj_value_loss \
  data=lotsa_v1_weighted \
  trainer.max_epochs=10_000 \
  train_dataloader.batch_size=128
```

# eval
```python
python -m cli.eval \
  run_name=example_base_moirai_small_wzj_value_loss \
  model=moirai_base_wzj_value_loss \
  model.patch_size=32 \
  model.context_length=1000 \
  data=etth1_test
```


```python
HYDRA_FULL_ERROR=1 python -m cli.eval \
  run_name=example_base_moirai_small_wzj_value_loss \
  model=moirai_base_wzj_value_loss \
  model.patch_size=32 \
  model.context_length=1000 \
  data=lsf_test \
  data.dataset_name=ETTh1 \
  data.prediction_length=96
```
