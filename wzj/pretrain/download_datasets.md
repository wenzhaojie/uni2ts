# 下载数据集

```shell
 huggingface-cli download Salesforce/lotsa_data --repo-type=dataset --local-dir /mnt/nas/Datasets/huggingface/lotsa_data
```

```shell
 huggingface-cli download autogluon/chronos_datasets \
  --repo-type dataset \
  --local-dir /mnt/nas/Datasets/huggingface/chronos_datasets
```

# 配置环境变量
```angular2html
echo "LOTSA_V1_PATH=/mnt/nas/Datasets/huggingface/lotsa_data" >> .env
```