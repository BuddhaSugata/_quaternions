import numpy as np
import matplotlib.pyplot as plt

# === Модель фотонного взаимодействия ===

# Параметры модели
N = 100  # количество источников
T = 10   # количество временных шагов
gamma = 1e-5  # коэффициент влияния потока на масштабный фактор

# Генерация случайных координат и светимостей источников
np.random.seed(42)
positions = np.random.uniform(-1, 1, (N, 3))
luminosities = np.random.uniform(0.5, 1.5, N)

# Расчёт попарного расстояния и потока
def compute_total_flux(positions, luminosities):
    flux = 0
    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(positions[i] - positions[j])
            if r > 1e-3:  # избегаем деления на 0
                flux += (luminosities[i] * luminosities[j]) / (r**2)
    return flux

# Вычисляем поток во времени (пока он фиксирован, можно усложнить позже)
flux_values = []
a_values = [1.0]  # начальный масштабный фактор
for t in range(T):
    flux = compute_total_flux(positions, luminosities)
    flux_values.append(flux)
    a_next = a_values[-1] * np.exp(gamma * flux)
    a_values.append(a_next)

# === Визуализация ===

fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axs[0].plot(range(T), flux_values, label="Photon Flux")
axs[0].set_ylabel("Total Photon Flux")
axs[0].grid()
axs[0].legend()

axs[1].plot(range(T + 1), a_values, label="Scale Factor a(t)", color="green")
axs[1].set_xlabel("Time Step")
axs[1].set_ylabel("a(t)")
axs[1].grid()
axs[1].legend()

plt.tight_layout()
plt.show()

# Генерируем затухающую во времени светимость: L_i(t) = L0_i * exp(-lambda * t)
lambda_decay = 0.1  # коэффициент затухания светимости
time = np.arange(T)

# Для каждого источника создадим траекторию L_i(t)
L0 = np.random.uniform(0.5, 1.5, N)  # начальные светимости
L_t = np.array([L0 * np.exp(-lambda_decay * t) for t in time])  # форма: (T, N)

# Считаем суммарную эмиссию и масштабный фактор
a_values_decay = [1.0]
total_emissions = []

for t in range(T):
    total_emission = np.sum(L_t[t])
    total_emissions.append(total_emission)
    a_next = a_values_decay[-1] * np.exp(gamma * total_emission)
    a_values_decay.append(a_next)

# Визуализация: эмиссия и масштабный фактор
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axs[0].plot(time, total_emissions, label="Total Emission L(t)", color="orange")
axs[0].set_ylabel("Total Emission")
axs[0].grid()
axs[0].legend()

axs[1].plot(range(T + 1), a_values_decay, label="a(t) from L(t)", color="purple")
axs[1].set_xlabel("Time Step")
axs[1].set_ylabel("Scale Factor a(t)")
axs[1].grid()
axs[1].legend()

plt.tight_layout()
plt.show()
