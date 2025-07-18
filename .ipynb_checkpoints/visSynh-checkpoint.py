import numpy as np
import matplotlib.pyplot as plt

# Цветовая карта (HSV → RGB)
def complex_to_rgb(z):
    absz = np.abs(z)
    angle = np.angle(z)
    hue = (angle + np.pi) / (2 * np.pi)
    saturation = np.ones_like(hue)
    value = 1 - 1 / (1 + absz**0.3)

    import colorsys
    rgb = np.vectorize(colorsys.hsv_to_rgb)(hue, saturation, value)
    return np.stack(rgb, axis=-1)

# Определение синтетической гиперболической функции
def synh(z):
    θ = np.real(z)
    σ = np.imag(z)
    return np.sinh(σ) * np.cos(θ) + 1j * np.cosh(σ) * np.sin(θ)

# Область определения
x = np.linspace(-4*np.pi, 4*np.pi, 800)
y = np.linspace(-10, 10, 800)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Значения функции
W_synh = synh(Z)

# Построение изображения
img = complex_to_rgb(W_synh)

# Визуализация
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', aspect='auto')
ax.set_title('Domain coloring of Synh(z)')
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')

# Контурная линия: Re(f(z)) = 1
real_part = np.real(W_synh)
contour = ax.contour(X, Y, real_part, levels=[1], colors='white', linewidths=1.0)
ax.clabel(contour, fmt='Re = 1', colors='white', fontsize=9)

plt.tight_layout()
plt.show()
