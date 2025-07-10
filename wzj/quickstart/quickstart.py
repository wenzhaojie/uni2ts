import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split

from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

# MODEL = "moirai-moe"  # model name: choose from {'moirai', 'moirai-moe'}
MODEL= "moirai"  # model name: choose from {'moirai', 'moirai-moe'}
SIZE = "small"  # model size: choose from {'small', 'base', 'large'}
PDT = 20  # prediction length: any positive integer
CTX = 200  # context length: any positive integer
PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 32  # batch size: any positive integer
TEST = 100  # test set length: any positive integer

# Read data into pandas DataFrame
url = (
    "https://gist.githubusercontent.com/rsnirwan/c8c8654a98350fadd229b00167174ec4"
    "/raw/a42101c7786d4bc7695228a0f2c8cea41340e18f/ts_wide.csv"
)
df = pd.read_csv(url, index_col=0, parse_dates=True)
# 导出csv
df.to_csv("ts_wide.csv", index=True)

# Convert into GluonTS dataset
ds = PandasDataset(dict(df))
print(f"数据集包含 {len(ds)} 个时间序列。")


# Split into train/test set
train, test_template = split(
    ds, offset=-TEST
)  # assign last TEST time steps as test set
print(f"训练集包含 {len(train)} 个时间序列")

# Construct rolling window evaluation
test_data = test_template.generate_instances(
    prediction_length=PDT,  # number of time steps for each prediction
    windows=TEST // PDT,  # number of windows in rolling window evaluation
    distance=PDT,  # number of time steps between each window - distance=PDT for non-overlapping windows
)
print(f"测试集包含 {len(test_data.input)} 个时间序列实例。")

# Prepare pre-trained model by downloading model weights from huggingface hub
if MODEL == "moirai":
    print(f"开始加载模型：{MODEL}，大小：{SIZE}，预测长度：{PDT}，上下文长度：{CTX}")
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{SIZE}"),
        prediction_length=PDT,
        context_length=CTX,
        patch_size=PSZ,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )
elif MODEL == "moirai-moe":
    print(f"开始加载模型：{MODEL}，大小：{SIZE}，预测长度：{PDT}，上下文长度：{CTX}")
    model = MoiraiMoEForecast(
        module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{SIZE}"),
        prediction_length=PDT,
        context_length=CTX,
        patch_size=16,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )

else:
    raise ValueError(f"Unknown model: {MODEL}")

print("模型加载完成。")
predictor = model.create_predictor(batch_size=BSZ)
print("开始预测...")
forecasts = list(predictor.predict(test_data.input))
print("预测完成。")

input_it = iter(test_data.input)
label_it = iter(test_data.label)
forecast_it = iter(forecasts)

inp = next(input_it)
label = next(label_it)
forecast = next(forecast_it)

plot_single(
    inp,
    label,
    forecast,
    context_length=200,
    name="pred",
    show_label=True,
)
plt.show()
plt.savefig("pred.png", dpi=300, bbox_inches="tight")