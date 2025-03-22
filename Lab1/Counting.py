import numpy as np
import matplotlib.pyplot as plt

a, b = 0, 1
x = np.linspace(-0.5, 1.5, 400)

pdf = np.where((x >= a) & (x <= b), 1, 0)  # Плотность вероятности
cdf = np.where(x < a, 0, np.where(x > b, 1, (x - a) / (b - a)))  # Функция распределения

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(x, pdf, color='blue', lw=2)
ax[0].fill_between(x, pdf, alpha=0.3, color='blue')
ax[0].set_title('Плотность вероятности f(x)')
ax[0].set_xlabel('x')
ax[0].set_ylabel('f(x)')
ax[0].grid()

ax[1].plot(x, cdf, color='red', lw=2)
ax[1].set_title('Функция распределения F(x)')
ax[1].set_xlabel('x')
ax[1].set_ylabel('F(x)')
ax[1].grid()

plt.savefig('uniform_distribution.png', dpi=300)
plt.show()
