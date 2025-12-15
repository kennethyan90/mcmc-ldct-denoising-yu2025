# MCMC-LDCT Redução de Ruído (Yu, 2025)

Reprodução do artigo Yu (2025) para disciplina de Estatística Computacional – PPCOMP/UFAPE.

**Artigo original**:  
Yu, S. Q. (2025). Markov Chain Monte Carlo-Based L₁/L₂ Regularization and Its Applications in Low-Dose CT Denoising. Journal of Applied Mathematics and Physics, 13, 419-428.  
DOI: https://doi.org/10.4236/jamp.2025.132021

**Autor da reprodução**: Kenneth Yan Santana Oliveira

## Como executar
1. Clone o repositório  
2. Instale dependências: `pip install numpy matplotlib`  
3. Rode: `python mcmc_l1l2_ldct_denoising.py`

Gera figuras com PSNR ≈ 25-26 dB e taxa de aceitação ≈ 0.50.

Resultados incluem variações de dose, anatomia e tamanho de passo.
