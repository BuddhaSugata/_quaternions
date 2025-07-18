import numpy as np
import matplotlib.pyplot as plt

# Цветовая карта: arg → цвет, abs → яркость
def complex_to_rgb(z):
    absz = np.abs(z)
    angle = np.angle(z)
    hue = (angle + np.pi) / (2 * np.pi)
    saturation = np.ones_like(hue)
    value = 1 - 1 / (1 + absz**0.3)
    
    import colorsys
    rgb = np.vectorize(colorsys.hsv_to_rgb)(hue, saturation, value)
    return np.stack(rgb, axis=-1)

# Новые синтетические гиперболические функции (без мнимой единицы)
def cosh_synthetic(z):
    θ = np.real(z)
    σ = np.imag(z)
    return np.cosh(σ) * np.cos(θ) + np.sinh(σ) * np.sin(θ)

def synh_synthetic(z):
    θ = np.real(z)
    σ = np.imag(z)
    return np.cosh(σ) * np.sin(θ) + np.sinh(σ) * np.cos(θ)

# Область комплексной плоскости
x = np.linspace(-2*np.pi, 2*np.pi, 800)
y = np.linspace(-3, 3, 800)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Вычисление функции f(z) = Cosh + Synh
W = cosh_synthetic(Z) + synh_synthetic(Z)

# Domain coloring
img = complex_to_rgb(W)

# Отображение
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', aspect='auto')
ax.set_title('Domain coloring of Cosh(z) + Synh(z) = e^z')
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')

# Контур: Re(f(z)) = 1
real_part = np.real(W)
contour = ax.contour(X, Y, real_part, levels=[1], colors='white', linewidths=1.0)
ax.clabel(contour, fmt='Re = 1', colors='white', fontsize=9)

plt.tight_layout()
plt.show()
