# 示例

# 训练一个small的模型
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
  run_name=example_moirai_base_wzj_value_loss \
  model=moirai_base_wzj_value_loss \
  model.patch_size=32 \
  model.context_length=1000 \
  data=lsf_test \
  data.dataset_name=ETTh1 \
  data.prediction_length=96
```

```python
HYDRA_FULL_ERROR=1 python -m cli.eval \
  run_name=example_moirai_small_wzj_value_loss \
  model=moirai_small_wzj_value_loss \
  model.patch_size=32 \
  model.context_length=1000 \
  data=lsf_test \
  data.dataset_name=ETTh1 \
  data.prediction_length=96
```


```python
HYDRA_FULL_ERROR=1 python -m cli.eval \
  run_name=example_moirai_small \
  model=moirai_1.0_R_small \
  model.patch_size=32 \
  model.context_length=1000 \
  data=lsf_test \
  data.dataset_name=ETTh1 \
  data.prediction_length=96
```

```python
HYDRA_FULL_ERROR=1 python -m cli.eval \
  run_name=example_moirai_base \
  model=moirai_1.0_R_base \
  model.patch_size=32 \
  model.context_length=1000 \
  data=lsf_test \
  data.dataset_name=ETTh1 \
  data.prediction_length=96
```



# 结果对比
```shell
(uni2ts) wzj@X99:~/GitHubProjects/uni2ts$ HYDRA_FULL_ERROR=1 python -m cli.eval \
  run_name=example_moirai_base_wzj_value_loss \
  model=moirai_base_wzj_value_loss \
  model.patch_size=32 \
  model.context_length=1000 \
  data=lsf_test \
  data.dataset_name=ETTh1 \
  data.prediction_length=96
[2025-08-03 22:06:58,927][datasets][INFO] - PyTorch version 2.4.1 available.
[2025-08-03 22:06:58,928][datasets][INFO] - JAX version 0.6.1 available.
19495it [01:20, 240.73it/s]
      MSE[mean]  MSE[0.5]  MAE[0.5]  MASE[0.5]  MAPE[0.5]  sMAPE[0.5]       MSIS  RMSE[mean]  NRMSE[mean]   ND[0.5]  mean_weighted_sum_quantile_loss
None   0.731709  0.678317  0.489907   1.140388   8.848048    0.887955  12.694545      0.8554     1.074735  0.615525                         0.501248
```


# 训练small之后，eval对比一下开源权重的版本
## 先看一下开源权重的eval结果
```shell
HYDRA_FULL_ERROR=1 python -m cli.eval \
  run_name=example_moirai_small_open_source \
  model=moirai_1.0_R_small \
  model.patch_size=32 \
  model.context_length=1000 \
  data=lsf_test \
  data.dataset_name=ETTh1 \
  data.prediction_length=96

      MSE[mean]  MSE[0.5]  MAE[0.5]  MASE[0.5]  MAPE[0.5]  sMAPE[0.5]     MSIS  RMSE[mean]  NRMSE[mean]   ND[0.5]  mean_weighted_sum_quantile_loss  ValueOrientedNLL[event_weight=2.0,smooth=0.1]
None   0.446563   0.44115  0.416185   0.967821   8.688899     0.76553  7.96007    0.668253     0.839601  0.522899                         0.420676                                            0.0
  
```

## 再看一下自己训练1000步的eval结果
