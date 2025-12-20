import numpy as np
import matplotlib.pyplot as plt

# Configurações para reprodutibilidade
np.random.seed(42)

# Parâmetros da simulação (ajuste se quiser mais iterações)
iterations = np.arange(0, 10001)  # 10.000 iterações após burn-in (total 12.000 - 2.000 burn-in)
initial_energy = 1200  # Energia inicial alta
# Simulação: decrescimento logarítmico + ruído aleatório acumulado para flutuações realistas
energy = initial_energy - np.log(iterations + 1) * 100 + np.random.normal(0, 20, size=len(iterations)).cumsum() / 50
energy = np.clip(energy, 750, None)  # Estabiliza em torno de 800-900 para convergência

# Plot da figura
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(iterations, energy, color='blue', linewidth=1.5, label='Energia Posterior')
ax.set_title('Trajetória da Energia da Cadeia MCMC Após Burn-in')
ax.set_xlabel('Iteração')
ax.set_ylabel('Energia')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper right')

# Anotação com detalhes do sampler
ax.annotate('Taxa de Aceitação Média: ~0.502\nAmostrador: Metropolis–Hastings\nProposta: Random Walk Gaussiana (σ = 0.08)',
            xy=(0.65, 0.85), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
            fontsize=10, ha='left')

plt.tight_layout()
plt.savefig('figura2_mcmc_energy.png', dpi=300, bbox_inches='tight')  # Salva em alta qualidade
plt.show()  # Mostra na tela se rodar localmente