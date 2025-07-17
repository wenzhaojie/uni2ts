import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split

from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

# ===============================
# 1. 生成多条正弦波时间序列 + 多窗口均值特征
# ===============================

N = 5               # 时间序列数量
TOTAL_LEN = 500     # 每条长度
PERIOD = 50         # 正弦波周期
NOISE_STD = 0.2     # 噪声幅度
freq = "D"          # 日期频率
date_range = pd.date_range("2020-01-01", periods=TOTAL_LEN, freq=freq)

def generate_single_ts_with_ma_features(date_range, item_id):
    """
    构造一条时间序列及三种不同长度的moving average特征
    """
    t = np.arange(len(date_range))
    # 主信号
    sine = np.sin(2 * np.pi * t / PERIOD + np.pi/4 * item_id) + np.random.normal(0, NOISE_STD, len(t))
    # 三种窗口的moving avg
    ma7 = pd.Series(sine).rolling(7, min_periods=1).mean()     # 7天滑动均值
    ma14 = pd.Series(sine).rolling(14, min_periods=1).mean()   # 14天滑动均值
    ma30 = pd.Series(sine).rolling(30, min_periods=1).mean()   # 30天滑动均值

    # 构造DataFrame（必须包含 target 列）
    df = pd.DataFrame({
        "target": sine,
        "item_id": item_id,
        "ma7": ma7,
        "ma14": ma14,
        "ma30": ma30,
    }, index=date_range)
    return df

# 多条序列组成dict
multiple_ts = {i: generate_single_ts_with_ma_features(date_range, i) for i in range(N)}
multiple_ts = {str(i): df for i, df in multiple_ts.items()}
# ===============
# 2. 构造 PandasDataset
# ===============
dynamic_real_cols = ["ma7", "ma14", "ma30"]

ds = PandasDataset(
    multiple_ts,
    feat_dynamic_real=dynamic_real_cols,
    target="target",
    freq=freq,
)

# ============
# 3. 划分训练/测试
# ============
prediction_length = 20
context_length = 200
test_len = 100

train, test_template = split(ds, offset=-test_len)
test_data = test_template.generate_instances(
    prediction_length=prediction_length,
    windows=test_len // prediction_length,
    distance=prediction_length,
)

# ============
# 4. Moirai 推理
# ============
model = MoiraiForecast(
    module=MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small"),
    prediction_length=prediction_length,
    context_length=context_length,
    patch_size="auto",
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=len(dynamic_real_cols),  # 3
    past_feat_dynamic_real_dim=0,
)

predictor = model.create_predictor(batch_size=8)
forecasts = list(predictor.predict(test_data.input))

input_it = list(test_data.input)
label_it = list(test_data.label)
forecast_it = forecasts

# ============
# 5. 可视化
# ============
for i in range(2):
    inp = input_it[i]
    label = label_it[i]
    forecast = forecast_it[i]
    print(f"Plotting window {i}, input keys: {list(inp.keys())}")
    plot_single(
        inp,
        label,
        forecast,
        context_length=context_length,
        name="pred",
        show_label=True,
    )
    plt.savefig(f"moirai_multi_mafeat_output_{i}.png", dpi=300, bbox_inches='tight')
    print(f"Output saved as moirai_multi_mafeat_output_{i}.png")
    plt.clf()
