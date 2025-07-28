import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split

from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

# ==========================
# 1. 人工构造正弦波时间序列数据
# ==========================
TOTAL_LEN = 500  # 总序列长度
NOISE_STD = 0.1  # 噪声标准差
PERIOD = 50      # 正弦周期长度
t = np.arange(TOTAL_LEN)
sine_wave = np.sin(2 * np.pi * t / PERIOD) + np.random.normal(0, NOISE_STD, TOTAL_LEN)
df = pd.DataFrame({'sine': sine_wave}, index=pd.date_range("2020-01-01", periods=TOTAL_LEN, freq="D"))
ds = PandasDataset(dict(df))

# ========== 2. 预测任务参数 ==========
SIZE = "small"
PDT = 20
CTX = 200
PSZ = "auto"
BSZ = 32
TEST = 100

# ========== 3. Split train/test ==========
train, test_template = split(ds, offset=-TEST)
test_data = test_template.generate_instances(
    prediction_length=PDT,
    windows=TEST // PDT,
    distance=PDT,
)

# ========== 4. 加载本地训练好的 HF Checkpoint ==========
# 指定你本地权重的路径
HF_CKPT_PATH = "/home/wzj/GitHubProjects/uni2ts/outputs/pretrain/moirai_small_wzj_value_loss/lotsa_v1_weighted/moirai_small/HF_checkpoints/last"

print("Loading local Moirai model from:", HF_CKPT_PATH)
moirai_module = MoiraiModule.from_pretrained(HF_CKPT_PATH)

model = MoiraiForecast(
    module=moirai_module,
    prediction_length=PDT,
    context_length=CTX,
    patch_size=PSZ,
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=ds.num_feat_dynamic_real,
    past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
)

predictor = model.create_predictor(batch_size=BSZ)
forecasts = predictor.predict(test_data.input)

# ========== 5. 绘图展示 ==========
input_it = list(test_data.input)
label_it = list(test_data.label)
forecast_it = list(forecasts)

for idx in range(2):  # 画前两个
    inp = input_it[idx]
    label = label_it[idx]
    forecast = forecast_it[idx]

    plot_single(
        inp,
        label,
        forecast,
        context_length=CTX,
        name="pred",
        show_label=True,
    )
    plt.savefig(f"moirai_sinewave_quickstart_output_{idx}.png", dpi=300, bbox_inches='tight')
    print(f"Output saved as moirai_sinewave_quickstart_output_{idx}.png")
    plt.clf()
