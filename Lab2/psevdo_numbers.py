import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import pandas as pd

def generate_central_limit(cnt, m, s):
    n = 12
    y = np.zeros(cnt)
    for i in range(cnt):
        r_sum = sum(np.random.random() for _ in range(n))
        y[i] = m + s * (r_sum - n / 2)
    return y

def generate_box_muller(cnt, m, s):
    z = np.zeros(cnt)
    for i in range(0, cnt - 1, 2):
        r1, r2 = np.random.random(), np.random.random()
        z0 = math.cos(2 * math.pi * r1) * math.sqrt(-2 * math.log(r2))
        z1 = math.sin(2 * math.pi * r1) * math.sqrt(-2 * math.log(r2))
        z[i] = m + s * z0
        if i + 1 < cnt:
            z[i + 1] = m + s * z1
    return z

def generate_marsaglia(cnt, m, s):
    z = np.zeros(cnt)
    i = 0
    while i < cnt - 1:
        u, v = np.random.uniform(-1, 1, 2)
        s_uv = u**2 + v**2
        if 0 < s_uv < 1:
            factor = math.sqrt(-2 * math.log(s_uv) / s_uv)
            z[i] = m + s * u * factor
            if i + 1 < cnt:
                z[i + 1] = m + s * v * factor
            i += 2
    return z

def generate_builtin(cnt, m, s):
    return m + s * np.random.randn(cnt)

def plot_histograms(dfs):
    for df_name, df in dfs.items():
        plt.figure(figsize=(15, 10))
        for i, column in enumerate(df.columns):
            plt.subplot(2, 2, i + 1)
            plt.hist(df[column], bins=50)
            plt.title(f'{column} ({df_name})')
            plt.xlabel('x')
            plt.ylabel('Частота')
            plt.grid(True)
        plt.tight_layout()
        plt.show()

def get_efr(data):
    x = np.sort(data)
    n = len(data)
    y = np.arange(1, n + 1) / n
    return x, y

def plot_empirical_cdf(dfs):
    for df_name, df in dfs.items():
        plt.figure(figsize=(15, 10))
        for i, column in enumerate(df.columns):
            plt.subplot(2, 2, i + 1)
            x, y = get_efr(df[column])
            plt.plot(x, y)
            plt.title(f'ЭФР для {column} ({df_name})')
            plt.xlabel('x')
            plt.ylabel('F(x)')
            plt.grid(True)
            plt.xlim(min(x), max(x))
            plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

def plot_scatter_matrices(dfs):
    for df_name, df in dfs.items():
        num_cols = len(df.columns)
        fig, axes = plt.subplots(2, math.ceil(num_cols / 2), figsize=(10, 10))
        axes = axes.flatten()
        for i, column in enumerate(df.columns):
            ax = axes[i]
            data = df[column]
            x_coords = data[::2]
            y_coords = data[1::2]
            min_len = min(len(x_coords), len(y_coords))
            x_coords = x_coords[:min_len]
            y_coords = y_coords[:min_len]
            ax.scatter(x_coords, y_coords, s=10)
            ax.set_title(f'Точечный график {column} ({df_name})')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True)
        plt.tight_layout()
        plt.show()

def plot_qq(dfs):
    for df_name, df in dfs.items():
        plt.figure(figsize=(15, 10))
        for i, column in enumerate(df.columns):
            plt.subplot(2, 2, i + 1)
            stats.probplot(df[column], dist="norm", plot=plt)
            plt.title(f'QQ-график для {column} ({df_name})')
            plt.xlabel('Теоретические квантильные значения')
            plt.ylabel('Эмпирические квантильные значения')
            plt.grid(True)
        plt.tight_layout()
        plt.show()

dfs = {
    'N = 1000': pd.DataFrame({
        'ЦПТ': generate_central_limit(1000, 11, 1),
        'Преобразование Бокса-Мюллера': generate_box_muller(1000, 11, 1),
        'Полярный метод Марсальи': generate_marsaglia(1000, 11, 1),
        'randn(n)': generate_builtin(1000, 11, 1)
    }),
    'N = 5000': pd.DataFrame({
        'ЦПТ': generate_central_limit(5000, 11, 1),
        'Преобразование Бокса-Мюллера': generate_box_muller(5000, 11, 1),
        'Полярный метод Марсальи': generate_marsaglia(5000, 11, 1),
        'randn(n)': generate_builtin(5000, 11, 1)
    }),
    'N = 10000': pd.DataFrame({
        'ЦПТ': generate_central_limit(10000, 11, 1),
        'Преобразование Бокса-Мюллера': generate_box_muller(10000, 11, 1),
        'Полярный метод Марсальи': generate_marsaglia(10000, 11, 1),
        'randn(n)': generate_builtin(10000, 11, 1)
    })
}
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', '{:.6f}'.format)

#plot_histograms(dfs)
plot_empirical_cdf(dfs)
plot_scatter_matrices(dfs)
plot_qq(dfs)

#print(dfs.keys())
#print(dfs.get('N = 1000'))

results = {}

for df_name, df in dfs.items():
    mean_vals = df.mean()
    var_vals = df.var()
    std_vals = df.std()

    results[df_name] = pd.DataFrame({
        'Математическое ожидание': mean_vals,
        'Дисперсия': var_vals,
        'СКО': std_vals
    })

# Вывод результатов
for df_name, result in results.items():
    print(f"\nРезультаты для {df_name}:")
    print(result)