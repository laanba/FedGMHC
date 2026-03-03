import matplotlib.pyplot as plt

# 1. 准备数据
data = """
1,warmup,0.738253,0.181421,2.061454,2208.2
2,warmup,0.823499,0.283638,0.861216,7091.9
3,warmup,0.844935,0.310160,0.674354,1372.3
4,warmup,0.852987,0.322843,0.613006,581.0
5,warmup,0.856407,0.323697,0.564415,688.1
6,clustered,0.857404,0.329139,0.537482,363.2
7,clustered,0.862347,0.330941,0.513164,370.2
8,clustered,0.864649,0.332631,0.487380,342.4
9,clustered,0.865651,0.334751,0.466538,350.4
10,clustered,0.868168,0.336268,0.444154,362.6
11,clustered,0.870329,0.340975,0.430323,349.9
12,clustered,0.872695,0.346019,0.417070,334.3
13,clustered,0.875211,0.349687,0.407146,331.9
14,clustered,0.876030,0.352464,0.396448,328.7
15,clustered,0.877262,0.354268,0.386619,330.6
16,clustered,0.878057,0.354406,0.372099,331.5
17,clustered,0.879081,0.353207,0.363600,326.5
18,clustered,0.880083,0.351321,0.353390,329.5
19,clustered,0.881396,0.352820,0.343279,331.5
20,clustered,0.881794,0.354353,0.335556,334.5
21,clustered,0.882739,0.354743,0.329154,332.7
22,clustered,0.884122,0.356077,0.320450,329.6
23,clustered,0.883430,0.355945,0.321022,332.6
24,clustered,0.885118,0.358536,0.312412,329.8
25,clustered,0.885950,0.360011,0.299048,328.7
26,clustered,0.885986,0.360762,0.295209,330.0
27,clustered,0.886933,0.361529,0.297208,331.1
28,clustered,0.886846,0.362228,0.287231,332.3
29,clustered,0.887126,0.364123,0.281990,331.0
30,clustered,0.887920,0.365339,0.279663,335.7
31,clustered,0.888252,0.366678,0.273719,335.2
32,clustered,0.888534,0.368997,0.269728,328.4
33,clustered,0.889191,0.371575,0.263869,330.4
34,clustered,0.890295,0.373945,0.263799,332.8
35,clustered,0.890882,0.375850,0.257516,329.2
36,clustered,0.891300,0.377119,0.251916,327.6
37,clustered,0.891695,0.379298,0.248666,328.3
38,clustered,0.892501,0.382340,0.247024,329.9
39,clustered,0.892284,0.384530,0.239849,329.5
40,clustered,0.892748,0.383639,0.237396,337.7
41,clustered,0.893223,0.386993,0.238792,352.4
42,clustered,0.893609,0.385094,0.232138,331.0
43,clustered,0.893823,0.386094,0.230139,335.0
44,clustered,0.894206,0.386430,0.227799,332.7
45,clustered,0.894682,0.387490,0.222207,332.2
46,clustered,0.894375,0.387873,0.217896,330.4
47,clustered,0.895110,0.387716,0.217111,327.4
48,clustered,0.895258,0.390536,0.218584,331.8
49,clustered,0.895303,0.387400,0.211670,333.1
50,clustered,0.895489,0.390360,0.213413,350.9
51,clustered,0.895314,0.391631,0.211843,336.9
52,clustered,0.895793,0.390940,0.209638,335.3
53,clustered,0.895906,0.393050,0.204944,335.0
54,clustered,0.895934,0.390201,0.203955,331.0
55,clustered,0.896222,0.394128,0.195957,331.7
56,clustered,0.896459,0.394501,0.199880,322.0
57,clustered,0.896722,0.393745,0.195295,332.7
"""

# 解析数据
epochs = []
pixel_acc = []
mIoU = []
stages = []

for line in data.strip().split('\n'):
    parts = line.split(',')
    epochs.append(int(parts[0]))
    stages.append(parts[1])
    pixel_acc.append(float(parts[2]))
    mIoU.append(float(parts[3]))

# 2. 绘图设置
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
plt.style.use('seaborn-v0_8-whitegrid')  # 设置清爽的风格

# 查找 warmup 阶段的结束点
warmup_end = stages.count('warmup')


def plot_with_stage(ax, x, y, title, ylabel, color):
    ax.plot(x, y, color=color, linewidth=2, marker='o', markersize=4, label=ylabel)
    # 填充阶段背景
    ax.axvspan(1, warmup_end, color='gray', alpha=0.1, label='Warmup')
    ax.axvspan(warmup_end, max(x), color='green', alpha=0.05, label='Clustered')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend()


# 绘制左图：像素预测率
plot_with_stage(ax1, epochs, pixel_acc, 'Pixel Accuracy Over Epochs', 'Pixel Accuracy', '#1f77b4')

# 绘制右图：mIoU
plot_with_stage(ax2, epochs, mIoU, 'mIoU Over Epochs', 'mIoU Score', '#ff7f0e')

plt.tight_layout()
plt.show()