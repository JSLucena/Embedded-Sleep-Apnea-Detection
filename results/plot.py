import matplotlib.pyplot as plt
import numpy as np

# Data
models = ["Baseline", "QAT", "INT8", "QAT+CMSIS-NN"]

# Latency data (ms)
latency_pico = [266.34, 198.07, 197.93, 19.21]
latency_pico2 = [23.12, 98.49, 98.41, 4.79]

# Memory usage (B)
flash_pico = [231996, 188300, 188060, 209544]
ram_pico = [101160, 57464, 57224, 57432]
flash_pico2 = [220568, 176864, 176624, 209316]
ram_pico2 = [101932, 58228, 57988, 58228]

# Latency bar chart
x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Inference Latency
ax[0].bar(x - width/2, latency_pico, width, label='Pico')
ax[0].bar(x + width/2, latency_pico2, width, label='Pico 2')
ax[0].set_ylabel('Latency (ms)')
ax[0].set_title('Inference Latency by Model and Device')
ax[0].set_xticks(x)
ax[0].set_xticklabels(models)
ax[0].legend()
ax[0].grid(True, linestyle='--', alpha=0.5)

# Plot 2: Stacked Memory Usage
bar1 = ax[1].bar(x - width/2, flash_pico, width, label='FLASH (Pico)', color='tab:blue')
bar2 = ax[1].bar(x - width/2, ram_pico, width, bottom=flash_pico, label='RAM (Pico)', color='tab:cyan')

bar3 = ax[1].bar(x + width/2, flash_pico2, width, label='FLASH (Pico 2)', color='tab:orange')
bar4 = ax[1].bar(x + width/2, ram_pico2, width, bottom=flash_pico2, label='RAM (Pico 2)', color='tab:red')

ax[1].set_ylabel('Memory (Bytes)')
ax[1].set_title('Memory Usage by Model and Device')
ax[1].set_xticks(x)
ax[1].set_xticklabels(models)
ax[1].legend()
ax[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np

labels = ['Accuracy', 'AUC', 'Precision', 'Recall', 'Specificity','F1-Score']
models = {
    "Baseline": [0.715, 0.850, 0.403, 0.819, 0.688 ,0.540],
    "Q-Aware": [0.732, 0.847, 0.419, 0.807, 0.713 ,0.551],
    "INT8": [0.516, 0.806, 0.286, 0.918, 0.413 ,0.437]
}

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]  # Loop back

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

for name, stats in models.items():
    stats += stats[:1]
    ax.plot(angles, stats, label=name)
    ax.fill(angles, stats, alpha=0.1)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_ylim(0, 1)
ax.set_title("Model Performance Comparison", pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

methods = ["Baseline", "QAT", "INT8", "QAT+CMSIS-NN"]
pico_latency = [266.34, 198.07, 197.93, 19.21]
pico2_latency = [23.12, 98.49, 98.41, 4.79]

# Calculate relative speedup (Baseline / Model)
pico_speedup = [pico_latency[0]/x for x in pico_latency]
pico2_speedup = [pico2_latency[0]/x for x in pico2_latency]

x = np.arange(len(methods))
bar_width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - bar_width/2, pico_speedup, width=bar_width, label="Pico")
plt.bar(x + bar_width/2, pico2_speedup, width=bar_width, label="Pico 2")

plt.xticks(x, methods)
plt.ylabel("Relative Speedup")
plt.title("Inference Speedup Relative to Baseline")
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

models = ["Baseline", "QAT", "INT8", "QAT+CMSIS-NN"]
flash = [231996, 188300, 188060, 209544]
latency = [266.34, 198.07, 197.93, 19.21]

plt.figure(figsize=(8, 6))
plt.scatter(flash, latency, color='blue')

for i, model in enumerate(models):
    plt.text(flash[i] + 1000, latency[i], model)

plt.xlabel("Flash Usage (Bytes)")
plt.ylabel("Latency (ms)")
plt.title("Flash vs Inference Latency (Pico)")
plt.grid(True)
plt.tight_layout()
plt.show()
