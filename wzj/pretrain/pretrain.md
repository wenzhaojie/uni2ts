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
  train_dataloader.batch_size=256
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