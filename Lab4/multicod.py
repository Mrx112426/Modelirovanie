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
    df = df.applymap(lambda x: round(float(str(x).replace(',', '.')), 2))# для проверки ручной части, когда числа в формате xx,xxx
    variant_column = df[f'{(k - 1) % 10 + 1}']
else:
    df = pd.read_csv('myData.csv')
    variant_column = df[f'variant_{(k - 1) % 20 + 1}']

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
    data_range = max(data) - min(data)
    num_intervals = int(np.sqrt(len(data)))
    return data_range, num_intervals

def plot_histogram(data):
    count, bins, ignored = plt.hist(data, bins='auto', alpha=0.7, color='blue', edgecolor='black')

    plt.xticks(bins)

    # Подписи и заголовок
    plt.title('Гистограмма')
    plt.xlabel('Границы интервалов')
    plt.ylabel('Частота')

    plt.show()

def plot_frequency_polygon(data):
    count, bins = np.histogram(data, bins='auto')

    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Строим полигон частот
    plt.plot(bin_centers, count, marker='o', color='red', linestyle='-', linewidth=2)

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

# Вычисления
print("Выборочное среднее:", sample_mean(data))
print("Выборочная дисперсия:", sample_variance(data))
print("Среднеквадратическое отклонение:", std_deviation(data))
print("Мода:", mode(data))
print("Медиана:", median(data))
print("Коэффициент асимметрии:", skewness(data))
print("Коэффициент эксцесса (без коррекции):", excess_kurtosis(data))

data_range, num_intervals = range_and_intervals(data)
print(f"Размах выборки: {data_range}")
print(f"Количество интервалов: {num_intervals}")


plot_histogram(data)
plot_frequency_polygon(data)
empirical_distribution(data)
