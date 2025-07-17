import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split

from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

# ==========================
# 人工构造正弦波时间序列数据
# ==========================
TOTAL_LEN = 500  # 总序列长度
NOISE_STD = 0.1  # 噪声标准差
PERIOD = 50      # 正弦周期长度
t = np.arange(TOTAL_LEN)
sine_wave = np.sin(2 * np.pi * t / PERIOD) + np.random.normal(0, NOISE_STD, TOTAL_LEN)

# 转成DataFrame (和多变量数据结构保持一致)
df = pd.DataFrame({'sine': sine_wave}, index=pd.date_range("2020-01-01", periods=TOTAL_LEN, freq="D"))

# 转成GluonTS Dataset
ds = PandasDataset(dict(df))

# 预测任务参数
# MODEL = "moirai-moe"  # 也可以切换为 'moirai'
MODEL = "moirai"  # 也可以切换为 'moirai'
SIZE = "small"
PDT = 20
CTX = 200
PSZ = "auto"
BSZ = 32
TEST = 100

# Split train/test
train, test_template = split(ds, offset=-TEST)

# 构建滚动窗口评估
test_data = test_template.generate_instances(
    prediction_length=PDT,
    windows=TEST // PDT,
    distance=PDT,
)

# 选择模型
if MODEL == "moirai":
    print("Using Moirai model...")
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
    print("Using Moirai-MoE model...")
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
    raise ValueError(f"Unsupported model: {MODEL}. Choose from 'moirai' or 'moirai-moe'.")

predictor = model.create_predictor(batch_size=BSZ)
forecasts = predictor.predict(test_data.input)

input_it = list(test_data.input)
label_it = list(test_data.label)
forecast_it = list(forecasts)

inp = input_it[0]
label = label_it[0]
forecast = forecast_it[0]

plot_single(
    inp,
    label,
    forecast,
    context_length=CTX,
    name="pred",
    show_label=True,
)
plt.savefig(f"{MODEL}_sinewave_quickstart_output.png", dpi=300, bbox_inches='tight')
print(f"Output saved as {MODEL}_sinewave_quickstart_output.png")
# 清空plt
plt.clf()


inp = input_it[1]
label = label_it[1]
forecast = forecast_it[1]


plot_single(
    inp,
    label,
    forecast,
    context_length=CTX,
    name="pred",
    show_label=True,
)
plt.savefig(f"{MODEL}_sinewave_quickstart_output_1.png", dpi=300, bbox_inches='tight')
print(f"Output saved as {MODEL}_sinewave_quickstart_output_1.png")
# 清空plt
plt.clf()