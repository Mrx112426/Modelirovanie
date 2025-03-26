import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

N_values = [1000, 5000, 10000]
sigma = 2

expected_value = sigma * np.sqrt(np.pi / 2)
variance = (2 - np.pi / 2) * sigma ** 2
std_dev = np.sqrt(variance)

def z(data):
    return int(np.sqrt(len(data)))

def razmax(data):
    return np.max(data) - np.min(data)

def length_interval(data):
    return razmax(data) / z(data)

def sample_mean(data):
    return np.mean(data)

def sample_variance(data):
    return np.var(data, ddof=1)

def std_deviation(data):
    return np.std(data, ddof=1)

def mode(data):
    mode_val, _ = stats.mode(data, keepdims=True)
    return mode_val[0]

def median(data):
    return np.median(data)

def skewness(data):
    return stats.skew(data)

def excess_kurtosis(data):
    return stats.kurtosis(data, fisher=False)

for N in N_values:
    U = np.random.rand(N)
    X = sigma * np.sqrt(-2 * np.log(U))

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.hist(X, bins=30, density=True, alpha=0.6, color='b', label='Гистограмма')
    x_vals = np.linspace(0, max(X), 100)
    pdf_vals = (x_vals / sigma ** 2) * np.exp(-x_vals ** 2 / (2 * sigma ** 2))
    plt.plot(x_vals, pdf_vals, 'r-', label='Теоретическая плотность')
    plt.xlabel('Значение')
    plt.ylabel('Плотность вероятности')
    plt.title(f'Распределение Рэлея (N={N})')
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 2)
    sorted_X = np.sort(X)
    empirical_cdf = np.arange(1, N + 1) / N
    plt.plot(sorted_X, empirical_cdf, label='Эмпирическая функция распределения')
    cdf_vals = 1 - np.exp(-x_vals ** 2 / (2 * sigma ** 2))
    plt.plot(x_vals, cdf_vals, 'r-', label='Теоретическая функция')
    plt.xlabel('Значение')
    plt.ylabel('Функция распределения')
    plt.title(f'Эмпирическая и теоретическая ФР (N={N})')
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.scatter(X[:-1], X[1:], s=1, color='blue', alpha=0.5, label='Точки (X_n, X_n+1)')
    plt.xlabel('X_n')
    plt.ylabel('X_n+1')
    plt.title(f'Диаграмма рассеяния (N={N})')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    print(f'Для N = {N}:')
    print(f'Среднее: {sample_mean(X):.4f} (теор. {expected_value:.4f})')
    print(f'Дисперсия: {sample_variance(X):.4f} (теор. {variance:.4f})')
    print(f'СКО: {std_deviation(X):.4f} (теор. {std_dev:.4f})')
    print(f'Медиана: {median(X):.4f}')
    print(f'Мода: {mode(X):.4f}')
    print(f'Размах: {razmax(X):.4f}')
    print(f'Количество интервалов: {z(X)}')
    print(f'Длина интервала: {length_interval(X):.4f}')
    print(f'Коэффициент асимметрии: {skewness(X):.4f}')
    print(f'Коэффициент эксцесса (без коррекции): {excess_kurtosis(X):.4f}\n')

