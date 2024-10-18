import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'
import pandas as pd


# JONSWAP波谱模型
def jonswap_spectrum(w, Hs, Tp, gamma):
    g = 9.81  # 重力加速度
    wp = 1 / Tp
    sigma = np.where(w > wp, 0.09, 0.07)
    # 计算波谱密度函数
    S = 319.34 * (Hs ** 2 / ((Tp ** 4) * (w ** 5))) * (np.exp(-1948 / (Tp * w) ** 4)) * (gamma ** (np.exp(-(0.159 * w * Tp - 1) ** 2 / (2 * sigma ** 2))))
    return S

Hs = 3.6  # 有义波高（m）
Tp = 14.5  # 谱峰周期（s）
gamma = 3  # 谱峰因子
w1 = np.arange(0.1, 1.5, 0.001)  # 生成从0到1之间，步长为0.01的数组
w6 = np.array([0.692, 0.554, 0.462, 0.426, 0.413, 0.401, 0.396, 0.39, 0.385, 0.379, 0.369, 0.346, 0.326, 0.308, 0.277])
S = jonswap_spectrum(w1, Hs, Tp, gamma)
S6 = jonswap_spectrum(w6, Hs, Tp, gamma)
print("S6:",S6)
# 画出波谱图
plt.plot(w1, S, color='blue')
plt.title("西非百年一遇JONSWAP谱")
plt.xlabel('w(rad/s)')
plt.ylabel('Sw')
plt.ylim(0, 17)
plt.grid(False)
plt.savefig("西非百年一遇JONSWAP谱")
plt.show()
max_index = np.argmax(S)
max_w1 = w1[max_index]
print("S最大的w为：", max_w1)

w_min = 0.30
w_max = 0.51
delta_w = 0.01
w2 = np.arange(w_min, w_max, delta_w)
print(w2)
a = []
w_avg = []
for i in range(len(w2) - 1):
    w_avg.append((w2[i] + w2[i+1]) / 2)
    a.append(np.sqrt(2 * jonswap_spectrum(w_avg[i], Hs, Tp, gamma) * delta_w))
    print("第", i+1, "个规则波w=", w_avg[i], "的波辐为", a[i])




