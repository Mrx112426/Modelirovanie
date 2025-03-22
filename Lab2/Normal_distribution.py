import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def plot_normal_distribution(mean, std):
    x = np.linspace(mean - 4 * std, mean + 4 * std, 1000)
    pdf = stats.norm.pdf(x, mean, std)  # Плотность вероятности (PDF)
    cdf = stats.norm.cdf(x, mean, std)  # Функция распределения (CDF)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # График PDF
    axes[0].plot(x, pdf, label=f'({mean}, {std}²)', color='b')
    axes[0].fill_between(x, pdf, alpha=0.2, color='b')
    axes[0].set_title('Плотность вероятности')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Плотность')
    axes[0].legend()
    axes[0].grid()

    # График CDF
    axes[1].plot(x, cdf, label=f'({mean}, {std}²)', color='r')
    axes[1].set_title('Функция распределения')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Вероятность')
    axes[1].legend()
    axes[1].grid()

    plt.show()


# Пример использования
plot_normal_distribution(mean=11, std=1)
