import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# 读取Excel文件，从第1100行开始，到第3000行结束
df = pd.read_excel('横摇试验数据.xlsx',  sheet_name=12, skiprows=8000, nrows=5000)  # 从第1100行开始，共nrows行数据

# 提取时间和横摇角度数据
time_data = df.iloc[::1, 1].tolist()  # 第二列是时间数据，从索引1开始，
roll_data = df.iloc[::1, 2].tolist()  # 第三列是横摇角数据，从索引2开始

roll_data_average = sum(roll_data) / len(roll_data)
print("Roll data average:", roll_data_average)
# 给原始roll数据减去平均值
roll_data_adjusted = [roll - roll_data_average for roll in roll_data]

plt.plot(time_data, roll_data, marker='o', linestyle='-', markersize=1)
# 添加标题和标签
plt.title('Roll Angle vs. Time(All)')
plt.xlabel('Time (s)')
plt.ylabel('Roll Angle (°)')
# 显示网格线
plt.grid(True)
# 显示曲线图
plt.show()

# 将横摇角数据转换为NumPy数组
roll_data_array = np.array(roll_data_adjusted)
time_data_array = np.array(time_data)

# 找到峰值
peaks, _ = find_peaks(roll_data_array, distance=170)  # 设定峰值之间的最小距离为20个数据点
# 找到波谷，将横摇角取负值再找峰值，即可找到波谷
valleys, _ = find_peaks(-roll_data_array, distance=170)  # 设定波谷之间的最小距离为20个数据点
# Calculate the time differences between consecutive peaks or valleys
time_diff = np.diff(np.array(time_data)[peaks])

phi = []
for i in range(4000):
    if i in peaks or i in valleys:
        phi.append(abs(roll_data_array[i]))
print("横摇角幅值平均值：", np.mean(phi))
print("船模频率：", 2*np.pi/np.mean(time_diff))
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



