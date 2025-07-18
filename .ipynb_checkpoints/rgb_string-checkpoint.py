import matplotlib.pyplot as plt
import numpy as np

# Профиль энергии струны между кварками
x = np.linspace(0.1, 3, 300)
V = x  # линейная зависимость

plt.plot(x, V, label='Энергия взаимодействия кварков', color='crimson')
plt.xlabel('Расстояние между кварками')
plt.ylabel('Потенциальная энергия')
plt.title('Конфайнмент: энергия струны в сильном взаимодействии')
plt.grid(True)
plt.legend()
plt.show()
