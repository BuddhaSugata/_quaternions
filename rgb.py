import matplotlib.pyplot as plt
import numpy as np

# Цветовые векторы
r = np.array([1, 0])
g = np.array([-0.5, np.sqrt(3)/2])
b = np.array([-0.5, -np.sqrt(3)/2])

# График
plt.figure(figsize=(6, 6))
plt.quiver(0, 0, r[0], r[1], angles='xy', scale_units='xy', scale=1, color='red', label='Red')
plt.quiver(0, 0, g[0], g[1], angles='xy', scale_units='xy', scale=1, color='green', label='Green')
plt.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Blue')

# Сумма = 0
sum_vec = r + g + b
plt.quiver(0, 0, sum_vec[0], sum_vec[1], angles='xy', scale_units='xy', scale=1, color='gray', linestyle='--')

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.grid()
plt.gca().set_aspect('equal')
plt.legend()
plt.title("Цветовое пространство SU(3) (упрощённая 2D модель)")
plt.show()
