import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("Введите ваш вариант")
k = int(input())
print("Введите 1 если хотите проверить вариант сделанный руками, иначе введите 2")
p = int(input())
if p == 1:
    df = pd.read_csv('Варианты_практика_1.csv', sep=';')
    df = df.map(lambda x: round(float(str(x).replace(',', '.')), 2))  # Исправлено applymap на map
    variant_column = df[f'{(k - 1) % 10 + 1}']
else:
    df = pd.read_csv('myData.csv')
    variant_column = df[f'variant_{(k - 1) % 20 + 1}']

def z(data):
    return int(np.sqrt(len(data)))

def razmax(data):
    return np.max(data) - np.min(data)

def lengt_interval(data):
    return razmax(data) / z(data)

def sample_mean(data):
    return sum(data) / len(data)


def sample_variance(data):
    mean = sample_mean(data)
    return sum((x - mean) ** 2 for x in data) / (len(data) - 1)


def std_deviation(data):
    return np.sqrt(sample_variance(data))


def mode(data):
    mode_result = stats.mode(data)
    if isinstance(mode_result.mode, np.ndarray):
        return mode_result.mode[0]
    else:
        return mode_result.mode

def excess_kurtosis(data):
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    return (np.sum((data - mean) ** 4) / (n * std_dev ** 4)) - 3


def median(data):
    return np.median(data)


def skewness(data):
    return stats.skew(data)


def range_and_intervals(data):
    data_min = min(data)
    data_max = max(data)
    data_range = data_max - data_min
    num_intervals = int(np.sqrt(len(data)))

    bin_width = data_range / num_intervals
    bins = [data_min + i * bin_width for i in range(num_intervals)]
    bins.append(data_max + 0.0001)  # Небольшая поправка для включения максимума
    return bins


def plot_histogram(data):
    bins = range_and_intervals(data)
    count, bins, patches = plt.hist(data, bins=bins, alpha=0.7, color='blue', edgecolor='black')


    if p == 1:
        plt.xticks(bins)
        plt.yticks(count)

    plt.title('Гистограмма')
    plt.xlabel('Границы интервалов')
    plt.ylabel('Частота')
    plt.show()
    return count, bins


def plot_frequency_polygon(data):
    count, bins = plot_histogram(data)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    plt.plot(bin_centers, count, marker='o', color='red', linestyle='-', linewidth=2)
    if p == 1:
        plt.xticks(bin_centers)
        plt.yticks(count)
    plt.title('Полигон частот')
    plt.xlabel('Значения')
    plt.ylabel('Частота')
    plt.show()


def empirical_distribution(data):
    data_sorted = np.sort(data)
    y = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    plt.step(data_sorted, y, where='post', color='green', linewidth=2)
    plt.title('Эмпирическая функция распределения')
    plt.xlabel('Значения')
    plt.ylabel('Проба')
    plt.show()


data = variant_column.values

print("Количество интервалов:", z(data))
print("Размах", razmax(data))
print("Длина интервала",lengt_interval(data))
print("Выборочное среднее:", sample_mean(data))
print("Выборочная дисперсия:", sample_variance(data))
print("Среднеквадратическое отклонение:", std_deviation(data))
print("Мода:", mode(data))
print("Медиана:", median(data))
print("Коэффициент асимметрии:", skewness(data))
print("Коэффициент эксцесса (без коррекции):", excess_kurtosis(data))

plot_frequency_polygon(data)
empirical_distribution(data)