import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split

from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

# ==========================
# 人工构造正弦波时间序列数据
# ==========================
TOTAL_LEN = 500
NOISE_STD = 0.1
PERIOD = 50
t = np.arange(TOTAL_LEN)
sine_wave = np.sin(2 * np.pi * t / PERIOD) + np.random.normal(0, NOISE_STD, TOTAL_LEN)
df = pd.DataFrame({'sine': sine_wave}, index=pd.date_range("2020-01-01", periods=TOTAL_LEN, freq="D"))
ds = PandasDataset(dict(df))

# ========== 参数设置 ==========
LOCAL_MODEL_CKPT = "/home/wzj/GitHubProjects/uni2ts/outputs/pretrain/moirai_small_wzj_value_loss/lotsa_v1_weighted/moirai_small/checkpoints/last.ckpt"  # <<< 改成你的路径
PDT = 20
CTX = 200
PSZ = "auto"
BSZ = 32
TEST = 100

# ========== 构建数据 ==========
train, test_template = split(ds, offset=-TEST)

test_data = test_template.generate_instances(
    prediction_length=PDT,
    windows=TEST // PDT,
    distance=PDT,
)

# ========== 加载本地 Moirai 模型 ==========
print(f"Loading local Moirai model from: {LOCAL_MODEL_CKPT}")
module = MoiraiModule.from_pretrained(
    pretrained_model_name_or_path=LOCAL_MODEL_CKPT,
    local_files_only=True,
)
model = MoiraiForecast(
    module=module,
    prediction_length=PDT,
    context_length=CTX,
    patch_size=PSZ,
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=ds.num_feat_dynamic_real,
    past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
)

# ========== 推理 ==========
predictor = model.create_predictor(batch_size=BSZ)
forecasts = predictor.predict(test_data.input)

input_it = list(test_data.input)
label_it = list(test_data.label)
forecast_it = list(forecasts)

# ========== 可视化 ==========
for i in range(2):  # 画前两个预测窗口
    inp = input_it[i]
    label = label_it[i]
    forecast = forecast_it[i]

    plot_single(
        inp,
        label,
        forecast,
        context_length=CTX,
        name=f"pred_{i}",
        show_label=True,
    )
    out_path = f"moirai_sinewave_local_last_1000_epoch_output_{i}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_path}")
    plt.clf()
