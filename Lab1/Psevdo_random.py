import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
N_values = [1000, 5000, 10000]

# 1. Мультипликативный конгруэнтный генератор
A = 7 ** 5
C = 0
M = 2 ** 32 - 1
R0 = 2 ** -52

def multiplicative_congruential(N):
    R = np.zeros(N)
    R[0] = R0
    for i in range(1, N):
        R[i] = (A * (R[i - 1]) + C) % M
    return R / M

# 2. Генератор Фибоначчи с запаздыванием
a, b = 63, 31

def fibonacci_generator(N):
    R = np.random.rand(max(a, b)).tolist()
    for i in range(max(a, b), N):
        R.append((R[i - a] - R[i - b]) % 1)
    return np.array(R)

# 3. Вихрь Мерсенна
def mersenne_twister(N):
    return np.random.rand(N)

methods = {
    "congruential": multiplicative_congruential,
    "fibonacci": fibonacci_generator,
    "mersenne": mersenne_twister
}
method_names = {
    "congruential": "Мультипликативный конгруэнтный генератор",
    "fibonacci": "Генератор Фибоначчи с запаздыванием",
    "mersenne": "Вихрь Мерсенна"
}

statistics = []

for N in N_values:
    for method, func in methods.items():
        data = func(N)
        txt_file = f'{method}_{N}.txt'
        png_file = f'{method}_{N}.png'
        if os.path.exists(txt_file):
            os.remove(txt_file)
        if os.path.exists(png_file):
            os.remove(png_file)
        np.savetxt(txt_file, data, fmt='%.10f')

        mean_value = np.mean(data)
        variance = np.var(data)
        std_dev = np.std(data)

        statistics.append([method_names[method], N, mean_value, variance, std_dev])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].hist(data, bins=30, alpha=0.75, color='blue')
        axes[0].set_title('Гистограмма')
        axes[0].set_xlabel('Значение')
        axes[0].set_ylabel('Частота')

        sorted_data = np.sort(data)
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        axes[1].plot(sorted_data, y, color='blue')
        axes[1].set_title('Эмпирическая функция распределения (CDF)')
        axes[1].set_xlabel('Значение')
        axes[1].set_ylabel('F(x)')

        axes[2].scatter(range(len(data)), data, s=1, color='blue')
        axes[2].set_title('Точечный график')
        axes[2].set_xlabel('Индекс')
        axes[2].set_ylabel('Значение')

        plt.suptitle(f"{method_names[method]} (N={N})")
        plt.savefig(png_file)
        plt.close()

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', '{:.6f}'.format)
df_statistics = pd.DataFrame(statistics, columns=["Метод", "N", "Матема. Ож", "Дисперсия", "СКО"])
df_statistics.to_csv("statistics.csv", index=False)
print(df_statistics)
