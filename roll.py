import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取Excel文件，从第1100行开始，到第3000行结束
df0 = pd.read_excel('横摇试验数据.xlsx',  sheet_name=0, skiprows=12, nrows=9600)  # 从第1100行开始，共nrows行数据
df = pd.read_excel('横摇试验数据.xlsx',  sheet_name=0, skiprows=600, nrows=2100)  # 从第1100行开始，共nrows行数据

# 提取时间和横摇角度数据
time_data = df.iloc[::1, 1].tolist()  # 第二列是时间数据，从索引1开始，每隔10行取一个数据
roll_data = df.iloc[::1, 2].tolist()  # 第三列是横摇角数据，从索引2开始，每隔10行取一个数据
time_data0 = df0.iloc[::1, 1].tolist()
roll_data0 = df0.iloc[::1, 2].tolist()

roll_data_average = sum(roll_data) / len(roll_data)
print("Roll data average:", roll_data_average)
# 给原始roll数据减去平均值
roll_data_adjusted = [roll - roll_data_average for roll in roll_data]
# print(time_data)
# 创建曲线图
plt.plot(time_data0, roll_data0, marker='o', linestyle='-', markersize=1)

# 添加标题和标签
plt.title('Roll Angle vs. Time(All)')
plt.xlabel('Time (s)')
plt.ylabel('Roll Angle (°)')

# 显示网格线
plt.grid(True)

# 显示曲线图
plt.show()

from scipy.signal import find_peaks

# 将横摇角数据转换为NumPy数组
roll_data_array = np.array(roll_data_adjusted)
time_data_array = np.array(time_data)

# 找到峰值
peaks, _ = find_peaks(roll_data_array, distance=200)  # 设定峰值之间的最小距离为20个数据点
# 找到波谷，将横摇角取负值再找峰值，即可找到波谷
valleys, _ = find_peaks(-roll_data_array, distance=200)  # 设定波谷之间的最小距离为20个数据点

# Calculate the time differences between consecutive peaks or valleys
time_diff = np.diff(np.array(time_data)[peaks])
# Calculate the average time difference (natural period)
natural_period = np.mean(time_diff)
print("Natural Period:", natural_period)
print("omega:", 2*np.pi/natural_period)

phi = []
for i in range(4000):
    if i in peaks or i in valleys:
        phi.append(abs(roll_data_array[i]))
print("phi数值：", phi)

# 在曲线图上标注峰值和波谷
plt.plot(time_data, roll_data_adjusted, marker='o', linestyle='-')
plt.plot(np.array(time_data)[peaks], np.array(roll_data_adjusted)[peaks], "x", color='red', label='peak')
plt.plot(np.array(time_data)[valleys], np.array(roll_data_adjusted)[valleys], "o", color='green', label='valley')
plt.title('Roll Angle vs. Time with Peaks and Valleys')
plt.xlabel('Time (s)')
plt.ylabel('Roll Angle (°)')
plt.legend()
plt.grid(True)
plt.show()

phi_ave=[]
phi_minus=[]
for i in range(len(phi)-5):
    phi_ave.append((phi[i+5]+phi[i])/2)
    phi_minus.append((phi[i]-phi[i+5])/phi_ave[-1])

print("\u03C6Am", phi_ave)
print("\u0394\u03C6A/\u03C6Am", phi_minus)
# plt.scatter(phi_ave, phi_minus)

# # 添加标题和标签
# plt.title('Roll Angle vs. Time')
# plt.xlabel('avg')
# plt.ylabel('minus/avg')

# # 显示网格线
# plt.grid(True)

# # 显示曲线图
# plt.show()

# 多项式拟合
coefficients = np.polyfit(phi_ave, phi_minus, 1)  # 一次多项式拟合
p = np.poly1d(coefficients)  # 创建拟合函数

# 绘制拟合曲线
plt.scatter(phi_ave, phi_minus, label='Data')
plt.plot(phi_ave, p(phi_ave), color='red', label='Fit')

# 添加标题和标签
plt.title('Linear Fit')
plt.xlabel('\u03C6Am')
plt.ylabel('\u0394\u03C6A/\u03C6Am')

# 显示图例
plt.legend()

# 显示网格线
plt.grid(True)

# 显示图形
plt.show()

# 输出拟合的斜率和截距
slope, intercept = coefficients
print("Slope-b:", slope)
print("Intercept-a:", intercept)
