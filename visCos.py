import numpy as np
import matplotlib.pyplot as plt

# HSV to RGB конвертер
def complex_to_rgb(z):
    absz = np.abs(z)
    angle = np.angle(z)

    hue = (angle + np.pi) / (2 * np.pi)
    saturation = np.ones_like(hue)
    value = 1 - 1 / (1 + absz**0.3)

    import colorsys
    rgb = np.vectorize(colorsys.hsv_to_rgb)(hue, saturation, value)
    return np.stack(rgb, axis=-1)

# Область определения: комплексная плоскость
x = np.linspace(-2*np.pi, 2*np.pi, 800)
y = np.linspace(-10, 10, 800)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# === СИНТЕТИЧЕСКИЕ ФУНКЦИИ ===
def syn(z):
    θ = np.real(z)
    σ = np.imag(z)
    return np.cosh(σ) * np.sin(θ) - 1j * np.sinh(σ) * np.cos(θ)

def cos(z):
    θ = np.real(z)
    σ = np.imag(z)
    return np.cosh(σ) * np.cos(θ) + 1j * np.sinh(σ) * np.sin(θ)

# === ВЫЧИСЛЕНИЯ ===
W_cos = cos(Z)
W_sum = cos(Z) + 1j * syn(Z)

# === ВИЗУАЛИЗАЦИЯ ===
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Cos(z)
img1 = complex_to_rgb(W_cos)
axes[0].imshow(img1, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', aspect='auto')
axes[0].set_title('Domain coloring of Cos(z)')
axes[0].set_xlabel('Re(z)')
axes[0].set_ylabel('Im(z)')

# Cos(z) + i·Syn(z)
img2 = complex_to_rgb(W_sum)
axes[1].imshow(img2, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', aspect='auto')
axes[1].set_title('Domain coloring of Cos(z) + i·Syn(z)')
axes[1].set_xlabel('Re(z)')
axes[1].set_ylabel('Im(z)')

plt.tight_layout()
plt.show()
