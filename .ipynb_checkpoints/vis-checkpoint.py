import numpy as np
import matplotlib.pyplot as plt

# Цветовая карта: аргумент и модуль
def complex_to_rgb(z):
    absz = np.abs(z)
    angle = np.angle(z)

    hue = (angle + np.pi) / (2 * np.pi)
    saturation = np.ones_like(hue)
    value = 1 - 1 / (1 + absz**0.3)

    import colorsys
    rgb = np.vectorize(colorsys.hsv_to_rgb)(hue, saturation, value)
    return np.stack(rgb, axis=-1)

# Синтетические функции
def syn(z):
    θ = np.real(z)
    σ = np.imag(z)
    return np.cosh(σ) * np.sin(θ) - 1j * np.sinh(σ) * np.cos(θ)

def cos(z):
    θ = np.real(z)
    σ = np.imag(z)
    return np.cosh(σ) * np.cos(θ) + 1j * np.sinh(σ) * np.sin(θ)

# Область
x = np.linspace(-4*np.pi, 4*np.pi, 800)
y = np.linspace(-5, 5, 800)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Функции
W_cos = cos(Z)
W_sum = cos(Z) + 1j * syn(Z)

# Цвета
img1 = complex_to_rgb(W_cos)
img2 = complex_to_rgb(W_sum)

# Визуализация
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# --- Cos(z)
axes[0].imshow(img1, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', aspect='auto')
axes[0].set_title('Domain coloring of Cos(z)')
axes[0].set_xlabel('Re(z)')
axes[0].set_ylabel('Im(z)')

# --- Cos(z) + i·Syn(z)
axes[1].imshow(img2, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', aspect='auto')
axes[1].set_title('Domain coloring of Cos(z) + i·Syn(z)')
axes[1].set_xlabel('Re(z)')
axes[1].set_ylabel('Im(z)')

# Добавляем контур: Re(f(z)) = 1
real_part = np.real(W_sum)
contour = axes[1].contour(X, Y, real_part, levels=[1], colors='white', linewidths=1.0)
axes[1].clabel(contour, fmt='Re = 1', colors='white', fontsize=9)

plt.tight_layout()
plt.show()
