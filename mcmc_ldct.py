import numpy as np
import matplotlib.pyplot as plt
import os

# Cria pasta para as figuras
if not os.path.exists('figuras'):
    os.makedirs('figuras')

np.random.seed(42)
n = 128

def gerar_e_salvar(dose_factor, hu_min, hu_max, sigma_prop, nome_arquivo, titulo):
    true = np.random.rand(n, n) * (hu_max - hu_min) + hu_min
    noisy = np.random.poisson(true * dose_factor) / dose_factor + np.random.normal(0, 10, true.shape)

    def log_posterior(x, y, lam=0.1, sigma2=100):
        return -0.5/sigma2 * np.sum((y-x)**2) - lam * (np.abs(x).sum() + 0.5*np.sum(x**2))

    N_iter, burnin = 12000, 2000
    x = noisy.copy()
    for i in range(N_iter):
        proposal = x + np.random.normal(0, sigma_prop, x.shape)
        if np.log(np.random.rand()) < log_posterior(proposal, noisy) - log_posterior(x, noisy):
            x = proposal
        if i >= burnin:
            if i == burnin: samples = [x.copy()]
            else: samples.append(x.copy())

    denoised = np.mean(samples, axis=0)
    psnr_val = 20 * np.log10(1000 / np.sqrt(np.mean((true - denoised)**2)))

    plt.figure(figsize=(15,4))
    plt.subplot(131); plt.imshow(true, cmap='gray', vmin=0, vmax=1500); plt.title('Ground Truth')
    plt.subplot(132); plt.imshow(noisy, cmap='gray', vmin=0, vmax=1500); plt.title(f'LDCT (dose {dose_factor*100:.0f}%)')
    plt.subplot(133); plt.imshow(denoised, cmap='gray', vmin=0, vmax=1500)
    plt.title(f'Denoised – PSNR = {psnr_val:.2f} dB')
    plt.suptitle(titulo, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'figuras/{nome_arquivo}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figura salva: {nome_arquivo}.png → PSNR = {psnr_val:.2f} dB")

# === AS 4 FIGURAS ===
gerar_e_salvar(0.20,   0, 1000, 0.08, "fig1_baseline",      "Baseline – dose 20%")
gerar_e_salvar(0.10,   0, 1000, 0.08, "fig2_dose10",        "Pergunta 1 – dose 10%")
gerar_e_salvar(0.20, 300, 1500, 0.08, "fig3_anatomia",      "Pergunta 2 – anatomia complexa")
gerar_e_salvar(0.20,   0, 1000, 0.16, "fig4_passo016",      "Pergunta 3 – σ_prop = 0.16")