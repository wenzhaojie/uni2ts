import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split

from uni2ts.model.moirai import MoiraiModule, MoiraiForecast
from uni2ts.distribution import MixtureOutput, StudentTOutput, NormalFixedScaleOutput, NegativeBinomialOutput, LogNormalOutput
from uni2ts.eval_util.plot import plot_single

# ========== 1. 还原模型参数 ==========
distr_output = MixtureOutput([
    StudentTOutput(),
    NormalFixedScaleOutput(),
    NegativeBinomialOutput(),
    LogNormalOutput(),
])
d_model = 384
num_layers = 6
patch_sizes = (8, 16, 32, 64, 128)
max_seq_len = 512
attn_dropout_p = 0.0
dropout_p = 0.0
scaling = True

# ========== 2. 构造人工正弦波数据 ==========
TOTAL_LEN = 500
NOISE_STD = 0.1
PERIOD = 50
t = np.arange(TOTAL_LEN)
sine_wave = np.sin(2 * np.pi * t / PERIOD) + np.random.normal(0, NOISE_STD, TOTAL_LEN)
df = pd.DataFrame({'sine': sine_wave}, index=pd.date_range("2020-01-01", periods=TOTAL_LEN, freq="D"))
ds = PandasDataset(dict(df))

# ========== 3. 数据集切分 ==========
PDT = 20
CTX = 200
PSZ = "auto"
BSZ = 32
TEST = 100
train, test_template = split(ds, offset=-TEST)
test_data = test_template.generate_instances(
    prediction_length=PDT,
    windows=TEST // PDT,
    distance=PDT,
)

# ========== 4. 加载本地 Moirai 权重 ==========
LOCAL_MODEL_CKPT = "/home/wzj/GitHubProjects/uni2ts/outputs/pretrain/moirai_small_wzj_value_loss/lotsa_v1_weighted/moirai_small/checkpoints/last.ckpt"
print(f"Loading local Moirai model from: {LOCAL_MODEL_CKPT}")

# 手动实例化模型
module = MoiraiModule(
    distr_output=distr_output,
    d_model=d_model,
    num_layers=num_layers,
    patch_sizes=patch_sizes,
    max_seq_len=max_seq_len,
    attn_dropout_p=attn_dropout_p,
    dropout_p=dropout_p,
    scaling=scaling,
)

# 加载权重
ckpt = torch.load(LOCAL_MODEL_CKPT, map_location="cpu")
if "state_dict" in ckpt:
    module.load_state_dict(ckpt["state_dict"])
else:
    module.load_state_dict(ckpt)

# ========== 5. 封装 MoiraiForecast ==========
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

# ========== 6. 推理 ==========
predictor = model.create_predictor(batch_size=BSZ)
forecasts = predictor.predict(test_data.input)

input_it = list(test_data.input)
label_it = list(test_data.label)
forecast_it = list(forecasts)

# ========== 7. 可视化 ==========
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
    out_path = f"moirai_sinewave_local_last_ckpt_pred_{i}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_path}")
    plt.clf()
