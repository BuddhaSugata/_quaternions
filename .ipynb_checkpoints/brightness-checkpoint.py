import numpy as np
import matplotlib.pyplot as plt

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
