# Импорты
import numpy as np
import matplotlib.pyplot as plt

# Цветовая карта: аргумент z будет задавать цвет, модуль — яркость
def complex_to_rgb(z):
    absz = np.abs(z)
    angle = np.angle(z)

    # Цветовой круг: оттенок по аргументу
    hue = (angle + np.pi) / (2 * np.pi)  # 0 to 1
    saturation = np.ones_like(hue)
    value = 1 - 1 / (1 + absz**0.3)      # яркость по модулю

    # HSV to RGB
    import colorsys
    rgb = np.vectorize(colorsys.hsv_to_rgb)(hue, saturation, value)
    return np.stack(rgb, axis=-1)

# Синтетическая функция: Syn(z) = cosh(σ) * sin(θ) - i * sinh(σ) * cos(θ)
def syn(z):
    θ = np.real(z)
    σ = np.imag(z)
    return np.cosh(σ) * np.sin(θ) - 1j * np.sinh(σ) * np.cos(θ)

# Область комплексной плоскости
x = np.linspace(-4*np.pi, 4*np.pi, 800)
y = np.linspace(-10, 10, 800)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Вычисляем функцию
W = syn(Z)

# Цветовая карта
img = complex_to_rgb(W)

# Отображение
plt.figure(figsize=(8, 8))
plt.imshow(img, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', aspect='auto')
plt.xlabel('Re(z)')
plt.ylabel('Im(z)')
plt.title('Domain Coloring of Syn(z)')
plt.show()
