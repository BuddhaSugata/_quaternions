# %% [markdown]
# Униметрия — Прото-функция (Unimetry — Proto-function)
#
# Jupyter-friendly script ("percent" format). Откройте в Jupyter/VSCode: каждая ячейка разделена `# %%`.
#
# Формат: markdown-ячейки с русским текстом и английскими терминами в скобках при первом упоминании;
# кодовые ячейки рисуют три наглядные схемы (colorful diagrams) и сохраняют их в /mnt/data.

# %% [markdown]
# Introduction & Historical Note
#
# Мы вводим концепт, в котором время и пространство вторичны относительно изменения единого прото-параметра $\chi$.
# Вся динамика объекта представлена как поток (flow) в евклидовом прото-пространстве — Хора (Khôra).
#
# Краткая справка (Historical note).
# Платон в "Тимее" использует термин *khôra* (χορα) — receptacle, место, вместилище. Ниже — краткая цитата
# (перевод на русский) и ссылка на источник для дальнейшего чтения.
#
# > "There is also a third kind of entity, a receptacle (khôra), which receives all things and 'gives room' to them." — Plato, *Timaeus* (paraphrase).
#
# (Можно заменить или уточнить перевод при желании.)

# %% [markdown]
# Goals of this notebook
# 1. Переработать текст до конца раздела 3.2 (сохраняем нумерацию уравнений).
# 2. Добавить пояснения перед важными блоками (Idea: ...).
# 3. Построить 3 наглядные цветные схемы:
#    - Vector $(\tilde{S},\tilde{L})$ in Euclidean Khôra;
#    - Radius geometry and gamma-factor (visual link to $\gamma$);
#    - Projection of flows on internal and external axes.
# 4. Сгенерировать и сохранить картинки в `./figures/`.

# %% [markdown]
# Notebook structure / Структура блоков
# - Markdown: формализм и пояснения, LaTeX-уравнения.
# - Code: визуализации (matplotlib).

# %%
# Код: импорт библиотек и настройка
import os
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Arc

# создать папку для фигур
out_dir = Path('figures')
out_dir.mkdir(exist_ok=True)

# общие параметры стиля (минималистично, но цветные)
plt.rcParams.update({'figure.dpi': 150, 'font.size': 11})

# %% [markdown]
# Visualization 1 — Vector $(\tilde{S},\tilde{L})$ in Euclidean Khôra
# Idea: показать евклидову декомпозицию прото-скорости $\tilde{H}$ на компоненты
# $\tilde{S}$ (real/internal) и $\tilde{L}$ (imag/external). Наглядно — вектор и его проекции.

# %%
def draw_vector_SL(tildeS=1.0, tildeL=0.6, show=True, filename=None):
    fig, ax = plt.subplots(figsize=(5,5))
    H = math.hypot(tildeS, tildeL)

    # оси
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)

    # вектор H
    arr = FancyArrowPatch((0,0), (tildeS, tildeL), arrowstyle='->', mutation_scale=18,
                          linewidth=2, color='#1f77b4')
    ax.add_patch(arr)

    # проекции
    ax.plot([tildeS, tildeS], [0, tildeL], linestyle='--', color='#ff7f0e')
    ax.plot([0, tildeS], [tildeL, tildeL], linestyle='--', color='#2ca02c')

    # подписи
    ax.text(tildeS/2, -0.08, r'$\tilde{S}$ (internal / temporal)', ha='center')
    ax.text(-0.08, tildeL/2, r'$\tilde{L}$ (external / spatial)', va='center', rotation='vertical')
    ax.text(tildeS*0.5, tildeL*0.6, r'$\tilde{H}$', color='#1f77b4')

    # окружность нормы H
    circ = Circle((0,0), H, fill=False, linestyle=':', edgecolor='#9467bd')
    ax.add_patch(circ)

    ax.set_xlim(-0.2, max(tildeS,H)*1.2)
    ax.set_ylim(-0.2, max(tildeL,H)*1.2)
    ax.set_aspect('equal')
    ax.set_title('Proto-velocity vector $\\tilde{V} = (\\tilde{S},\\tilde{L})$ in Kh\^ora')
    ax.set_xlabel('real axis — temporal (internal)')
    ax.set_ylabel('imag axis — spatial (external)')
    ax.grid(False)

    plt.tight_layout()
    if filename:
        fig.savefig(filename)
    if show:
        plt.show()
    plt.close(fig)

# пример
fig1 = out_dir / 'vector_SL.png'
draw_vector_SL(tildeS=1.0, tildeL=0.6, filename=str(fig1))
print('Saved:', fig1)

# %% [markdown]
# Visualization 2 — Radius geometry and $\gamma$-factor
# Idea: показать, как радиус (R) и фазовая скорость $\dot{\phi}$ влияют на фактор
# $\breve{H} = (1 - R^2 \dot{\phi}^2 / \dot{H}^2)^{-1/2} \sim \gamma$.

# %%
def draw_gamma_geometry(R=1.0, phi_dot=0.6, H_dot=1.0, show=True, filename=None):
    # рассчитать effective gamma
    denom = 1 - (R**2 * phi_dot**2) / (H_dot**2)
    gamma = 1.0 / math.sqrt(denom) if denom>0 else float('inf')

    fig, ax = plt.subplots(figsize=(6,4))
    ax.set_title('Radius (R) and phase-speed influence on $\\breve{H} \sim \\gamma$')

    # график gamma vs R при фиксированном phi_dot
    Rs = np.linspace(0.01, 1.4*R, 300)
    gammas = []
    for r in Rs:
        d = 1 - (r**2 * phi_dot**2) / (H_dot**2)
        gammas.append(1.0/math.sqrt(d) if d>0 else np.nan)

    ax.plot(Rs, gammas, linewidth=2, color='#d62728')
    ax.axvline(R, linestyle='--', color='#2ca02c', label=f'R = {R}')
    ax.axhline(gamma, linestyle='--', color='#9467bd', label=f'gamma ~ {gamma:.2f}')

    ax.set_xlabel('R (proto-radius)')
    ax.set_ylabel('breve{H} ~ gamma')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if filename:
        fig.savefig(filename)
    if show:
        plt.show()
    plt.close(fig)

fig2 = out_dir / 'gamma_geometry.png'
draw_gamma_geometry(R=1.0, phi_dot=0.6, H_dot=1.0, filename=str(fig2))
print('Saved:', fig2)

# %% [markdown]
# Visualization 3 — Projection of flows on internal and external axes
# Idea: показать, как простые (photon-like) и сложные (massive-like) объекты располагаются
# в пространстве (\\tilde{S},\\tilde{L}) — простые вдоль мнимой оси, сложные вдоль реальной.

# %%
def draw_flow_projection(points=None, show=True, filename=None):
    if points is None:
        # набор точек: (tildeS, tildeL, label, color)
        points = [
            (0.0, 1.0, 'photon (simple)', '#17becf'),
            (1.0, 0.0, 'massive (complex)', '#e377c2'),
            (0.6, 0.8, 'observer (mixed)', '#1f77b4')
        ]
    fig, ax = plt.subplots(figsize=(6,6))
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)

    for s,l,label,color in points:
        ax.scatter(s, l, s=80, color=color)
        ax.text(s+0.03, l+0.03, label)
        ax.plot([s, s], [0, l], linestyle='--', color=color, alpha=0.5)
        ax.plot([0, s], [l, l], linestyle='--', color=color, alpha=0.5)

    # окружности для норм
    maxr = max(math.hypot(p[0], p[1]) for p in points) * 1.2
    for r in np.linspace(0.5, maxr, 3):
        circ = Circle((0,0), r, fill=False, linestyle=':', edgecolor='#7f7f7f')
        ax.add_patch(circ)

    ax.set_xlim(-0.3, maxr)
    ax.set_ylim(-0.3, maxr)
    ax.set_aspect('equal')
    ax.set_title('Projection of flows: simple vs complex objects')
    ax.set_xlabel('real — internal (\\tilde{S})')
    ax.set_ylabel('imag — external (\\tilde{L})')
    plt.tight_layout()
    if filename:
        fig.savefig(filename)
    if show:
        plt.show()
    plt.close(fig)

fig3 = out_dir / 'flow_projection.png'
draw_flow_projection(filename=str(fig3))
print('Saved:', fig3)

# %% [markdown]
# Далее в ноутбуке следует переработанный текст (markdown) до конца раздела 3.2.
# Я подготовлю аккуратно отформатированный блок Markdown с пояснениями "Idea:" перед важными уравнениями
# и поясняющими сносками. Это будет следующая markdown-ячейка в ноутбуке при экспорте.

# %% [markdown]
# ---
# Готово: сгенерированы три наглядные схемы и создана структура ноутбука.
# Файлы сохранены в `./figures/`:
# - vector_SL.png
# - gamma_geometry.png
# - flow_projection.png
#
# Хотите, чтобы я:
# 1) Экспортировал этот файл как полноценный `.ipynb` и положил его в корень (и дал ссылку на скачивание)?
# 2) Вставил сейчас переработанный Markdown (текст до конца 3.2) прямо в этот notebook (следующей ячейкой)?
# 3) Или сразу сделал PDF-версию статьи из этого notebook?
# 
# Выбирайте одну или несколько опций (ответьте коротко: 1, 2, 3 или комбинацию, например "1 и 2").
