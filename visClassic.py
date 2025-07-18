import numpy as np
import matplotlib.pyplot as plt

# Цветовая карта: arg → цвет, abs → яркость
def complex_to_rgb(z):
    absz = np.abs(z)
    angle = np.angle(z)
    hue = (angle + np.pi) / (2 * np.pi)      # от 0 до 1
    saturation = np.ones_like(hue)
    value = 1 - 1 / (1 + absz**0.3)           # логарифмический масштаб
    
    import colorsys
    rgb = np.vectorize(colorsys.hsv_to_rgb)(hue, saturation, value)
    return np.stack(rgb, axis=-1)

# Классические определения sin(z) и cos(z)
def sin_complex(z):
    return np.sin(z)

def cos_complex(z):
    return np.cos(z)

# Область z = x + i y
x = np.linspace(-2*np.pi, 2*np.pi, 800)
y = np.linspace(-3, 3, 800)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Визуализация
def plot_domain_coloring(f, title):
    W = f(Z)
    img = complex_to_rgb(W)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', aspect='auto')
    ax.set_title(f'Domain coloring of {title}')
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')

    # Линия уровня: Re(f(z)) = 0
    contour = ax.contour(X, Y, np.real(W), levels=[1], colors='white', linewidths=1.0)
    ax.clabel(contour, fmt='Re = 1', colors='white', fontsize=9)

    plt.tight_layout()
    plt.show()

# Визуализация классического cos(z)
plot_domain_coloring(cos_complex, "cos(z)")

# Визуализация классического sin(z)
plot_domain_coloring(sin_complex, "sin(z)")
