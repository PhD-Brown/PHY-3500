# 🔭 AstroSpectro · PHY-3500 — Réduction de dimension sur spectres stellaires

> **Réduction de dimension appliquée aux spectres stellaires LAMOST DR5 × Gaia DR3**  
> Comparaison de trois approches : PCA · UMAP · Autoencodeur neuronal

<div align="center">

[![Université Laval](https://img.shields.io/badge/Université_Laval-PHY--3500-red?style=for-the-badge)](https://www.ulaval.ca/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![LAMOST DR5](https://img.shields.io/badge/Data-LAMOST_DR5-purple?style=for-the-badge)](http://www.lamost.org/)

</div>

---

## 👥 Équipe

| Rôle | Nom |
|---|---|
| Conception & pipeline complet | **Alex Brown** |
| Rapport & présentation | **Justine Jean** |
| Rapport & présentation | **Nerimantas Caillat** |

**Cours :** PHY-3500 Physique numérique · Prof. Antoine Allard · Université Laval  
**Symposium écrit :** 24 avril 2026  
**Présentation orale :** 30 avril – 1er mai 2026 · Musée de la Civilisation de Québec

---

## 📋 Table des matières

1. [Aperçu du projet](#-aperçu-du-projet)
2. [Données](#-données)
3. [Structure du dépôt](#-structure-du-dépôt)
4. [Installation](#-installation)
5. [Résultats — PCA](#-résultats--pca-notebook-01)
6. [Résultats — UMAP & t-SNE](#-résultats--umap--t-sne-notebook-02)
7. [Résultats — Autoencodeur neuronal](#-résultats--autoencodeur-neuronal-notebook-03)
8. [Synthèse comparative](#-synthèse-comparative)
9. [Guide des figures](#-guide-des-figures)
10. [Livrables académiques](#-livrables-académiques)
11. [Références](#-références)

---

## 🌌 Aperçu du projet

Ce projet applique trois méthodes de réduction de dimension à **42 920 spectres stellaires** issus du relevé LAMOST DR5, croisés avec les paramètres astrophysiques de Gaia DR3.

### Question de recherche

> **Trois méthodes de réduction de dimension — linéaire (PCA), non-linéaire topologique (UMAP), non-linéaire neuronale (Autoencodeur) — capturent-elles la même physique stellaire lorsqu'on projette 208 features en 2 dimensions ?**

### Lien avec PHY-3500

La PCA repose sur la **décomposition spectrale** (valeurs propres, SVD tronquée) — thème central du cours. Ce projet en est l'application directe sur des données astrophysiques réelles.

---

## 📡 Données

### Source

| Paramètre | Valeur |
|---|---|
| Relevé source | LAMOST DR5 (Large Sky Area Multi-Object Fiber Spectroscopic Telescope) |
| Cross-match | Gaia DR3 (paramètres astrophysiques) |
| Spectres analysés | **42 920** |
| dont étoiles (STAR) | 42 862 · 99,9 % |
| dont galaxies (GALAXY) | 54 · 0,13 % |
| dont quasars (QSO) | 4 · 0,009 % |
| Filtrage qualité | SNR bande r ≥ 10 |
| Features spectroscopiques | **208 descripteurs engineerés** |

### Features engineerées (208D)

Les spectres bruts (~3 900 pixels) sont transformés en **208 descripteurs physiques** :

- Largeurs équivalentes (EW) des raies : Hα, Hβ, Hγ, Hδ, Hε, H8, Ca II H&K, Mg b, Fe 5270, Fe 5335...
- FWHM (Full Width at Half Maximum) des raies
- Indices de couleur synthétiques (B−V, U−B, g−r...)
- Pentes et asymétries du continuum
- Indices de métallicité (FeH_proxy, metal_index_combined, alpha_elements_index)
- Indices SNR par bande spectrale

### Paramètres Gaia DR3 disponibles

`T_eff` · `log g` · `[Fe/H]` · couleur `G_BP−G_RP` · parallaxe · magnitudes absolues

---

## 📁 Structure du dépôt

```
TP_final/
├── data/                          # Données (non versionnées — voir .gitignore)
│   ├── raw/                       # Spectres FITS LAMOST DR5 bruts
│   └── processed/                 # Features engineerées + catalogue Gaia
├── figs/                          # Toutes les figures générées
│   ├── pca_variance_explained.png
│   ├── pca_loadings_pc1.png
│   ├── pca_loadings_pc2.png
│   ├── pca_eigenspectra.png
│   ├── pca_correlation_heatmap.png
│   ├── pca_scores_grid.png
│   ├── umap_classes.png
│   ├── umap_negative_control.png
│   ├── stability_umap.png
│   ├── stability_tsne.png
│   ├── ae_training_history.png
│   ├── ae_latent_grid_zoomed.png
│   ├── ae_vs_pca_mse.png
│   ├── ae_error_distribution_logscale.png
│   ├── ae_latent_interpolation.png
│   └── synthesis_pca_umap_ae.png  # ← Figure centrale du projet
├── notebooks/
│   ├── phy3500_01_pca.ipynb       # Analyse PCA complète
│   ├── phy3500_02_umap_tsne.ipynb # UMAP, t-SNE, stabilité
│   └── phy3500_03_autoencoder.ipynb # Autoencodeur PyTorch
├── src/                           # Modules Python partagés
│   ├── pca_analyzer.py
│   ├── dimred_visualizer.py
│   ├── autoencoder.py             # SpectralAutoencoder (PyTorch)
│   ├── embedding.py
│   └── data_loader.py
├── results/                       # Rapports de runs (JSON)
├── requirements.txt
├── setup_venv.ps1
├── setup_venv.sh
└── README.md                      # Ce fichier
```

---

## ⚙️ Installation

### Prérequis

- Python 3.11+
- GPU recommandé pour l'autoencodeur (CUDA 12+) — CPU fonctionnel sinon
- ~4 Go de RAM minimum

### Mise en place

```bash
# Cloner le dépôt
git clone https://github.com/PhD-Brown/PHY-3500.git
cd PHY-3500/TP_final

# Linux / macOS
bash setup_venv.sh

# Windows (PowerShell)
./setup_venv.ps1
```

### Dépendances principales

```
scikit-learn>=1.3
umap-learn>=0.5
torch>=2.0
matplotlib>=3.7
seaborn>=0.12
pandas>=2.0
numpy>=1.25
scipy>=1.11
joblib>=1.3
```

### Exécution des notebooks

```bash
jupyter lab
# → ouvrir dans l'ordre : 01_pca → 02_umap_tsne → 03_autoencoder
```

> **Note :** Les fichiers de données (`data/`) ne sont pas versionnés. Contacter Alex pour accéder aux features pré-calculées.

---

## 📊 Résultats — PCA (Notebook 01)

### Variance expliquée

| Seuil | Features engineerées (208D) | Spectres bruts (~3 900 px) |
|---|---|---|
| 80 % | 51 PCs | 3 PCs |
| 90 % | 73 PCs | 3 PCs |
| 95 % | **91 PCs** | **5 PCs** |
| 99 % | 100 PCs | 36 PCs |
| Maximum (50 PCs) | 75,1 % | 99,2 % |

> **Interprétation :** Il faut 91 PCs pour capturer 95 % de la variance des features engineerées, contre seulement 5 PCs pour les spectres bruts. Les features physiques distribuent l'information de façon plus orthogonale — elles sont plus riches pour l'apprentissage automatique mais moins compressibles linéairement.

### Top 10 features — PC1 (16,9 % de variance)

| Feature | Loading | Interprétation physique |
|---|---|---|
| `Hα_eq_width` | +0.173 | Raie de Balmer : étoiles chaudes |
| `continuum_asymmetry` | +0.172 | Structure spectrale globale |
| `Mg_b_eq_width` | −0.171 | Raie de métallicité : étoiles froides |
| `synthetic_BV` | −0.163 | Couleur : étoiles froides/rouges |
| `flux_ratio_blue_red` | +0.161 | Étoiles chaudes (flux bleu dominant) |
| `continuum_slope_global` | −0.156 | Pente vers le rouge |
| `Hepsilon_eq_width` | −0.149 | Raie de Balmer |
| `H8_eq_width` | −0.148 | Raie de Balmer |
| `CaII_H_eq_width` | −0.147 | Ca II H : étoiles froides |
| `UV_excess_3900` | +0.144 | Excès UV : étoiles jeunes/chaudes |

### Top features — PC2 (11,9 % de variance)

| Feature | Loading | Interprétation physique |
|---|---|---|
| `FeH_proxy` | +0.205 | Indicateur de fer spectroscopique |
| `metal_index_combined` | +0.198 | Index combiné de métallicité |
| `alpha_elements_index` | +0.194 | Éléments alpha (Mg, Si, Ca) |
| `metal_poor_index` | −0.193 | Indicateur pauvreté en métaux |
| `Fe_5270_eq_width` | +0.174 | Raie de fer Fe 5270 |

### Corrélations Spearman PC ↔ paramètres Gaia

| Composante | T_eff | log g | [Fe/H] | G_BP−G_RP |
|---|---|---|---|---|
| **PC1** | **+0.832** | −0.157 | −0.546 | −0.765 |
| PC2 | +0.155 | +0.072 | −0.075 | −0.010 |
| PC3 | +0.234 | −0.408 | +0.063 | +0.115 |
| PC9 | −0.033 | −0.022 | +0.081 | −0.42 (RA) |
| PC10 | +0.039 | +0.048 | −0.116 | −0.46 (Dec) |

### Erreur de reconstruction vs nombre de PCs

| n PCs | MSE moyenne |
|---|---|
| 1 | 0.8351 |
| 2 | 0.7317 |
| 5 | 0.6290 |
| 10 | 0.5430 |
| 20 | 0.4289 |
| 50 | 0.2443 |

### 🔑 Résultat central NB01

> **PC1 (ρ = +0.832 avec T_eff) est l'axe de température stellaire.** La PCA non supervisée retrouve la séquence de Hertzsprung-Russell sans aucun label. PC2 est dominée par des indicateurs de métallicité spectroscopique (FeH_proxy, Fe_5270, Mg_5184) mais sa corrélation avec [Fe/H] Gaia est quasi nulle (ρ = −0.075) — elle corrèle plutôt avec le SNR (snr_g : −0.33), révélant une **contamination de la qualité observationnelle**.

### Note instrumentale — eigen-spectres sur flux bruts

Les premiers eigen-spectres présentent des **discontinuités à ~6 400 Å** — la jonction entre le bras bleu (3 700–5 900 Å) et le bras rouge (5 700–9 000 Å) du spectrographe LAMOST. Les premières PCs capturent donc un artefact instrumental, pas de la physique stellaire. C'est pourquoi le pipeline AstroSpectro utilise des features engineerées et non les flux bruts.

### Note sur les axes spatiaux

PC9 corrèle avec l'ascension droite (RA, ρ = −0.42) et PC10 avec la déclinaison (Dec, ρ = −0.46). Cela reflète des variations systématiques position-dépendantes dans LAMOST : gradients de calibration, populations galactiques (Population I vs II), et effets d'airmass selon la déclinaison.

---

## 🗺️ Résultats — UMAP & t-SNE (Notebook 02)

### Paramètres et performance

| Méthode | Temps (s) | Hyperparamètres | Stabilité Procrustes moy. |
|---|---|---|---|
| **UMAP** | 39.9 | n_neighbors=15, min_dist=0.1 | 0.036 → très stable (< 0.05) |
| **t-SNE** | 77.5 | perplexity=30, init=pca, max_iter=1000 | 0.00025 → quasi-déterministe |

> **Note :** UMAP est appliqué sur les 50 premières PCs (75,1 % de variance) plutôt que directement sur les 208 features. Ce choix réduit le bruit, accélère le calcul et stabilise les résultats.

### Stabilité UMAP — 5 seeds

| Seed | Distance Procrustes | Interprétation |
|---|---|---|
| 0 (référence) | 0.000 | — |
| 1 | 0.016 | Très stable |
| 2 | 0.044 | Très stable |
| 3 | 0.038 | Très stable |
| 4 | 0.045 | Très stable |

### Stabilité t-SNE — 5 seeds

| Seed | Distance Procrustes | Interprétation |
|---|---|---|
| 0 (référence) | 0.000 | — |
| 1 | 0.000442 | Quasi-identique |
| 2 | 0.000071 | Quasi-identique |
| 3 | 0.000112 | Quasi-identique |
| 4 | 0.000395 | Quasi-identique |

### 🔑 Résultat central NB02

> **UMAP révèle une structure filamentaire complexe** (séquence principale, géantes rouges, naines M, branches latérales) **invisible dans la projection PCA.** Ce contraste constitue l'argument empirique central pour la non-linéarité de l'espace spectral stellaire. Le **contrôle négatif** valide scientifiquement ces structures : des données permutées aléatoirement produisent un blob diffus et informe — les structures observées sont réelles.

---

## 🧠 Résultats — Autoencodeur neuronal (Notebook 03)

### Architecture

```
Encodeur :  208 → 256 → 128 → 64 → z(2)
Décodeur :  z(2) → 64 → 128 → 256 → 208
```

| Composant | Détail |
|---|---|
| Activations | ReLU + BatchNorm1d + Dropout(0.1) |
| Sortie | Linéaire (features standardisées) |
| Loss | MSE |
| Optimiseur | Adam + ReduceLROnPlateau |
| Early stopping | patience = 25 epochs |
| latent_dim | **2** (comparaison directe PCA/UMAP) |

### Métriques d'entraînement

| Paramètre | Valeur |
|---|---|
| Temps d'entraînement | 190.9 secondes |
| Epochs effectuées | 190 (best epoch : 165) |
| **Best val_loss (MSE)** | **0.51835** |
| Final train_loss | 0.56228 |
| Final val_loss | 0.51939 |
| MSE globale reconstruction | 0.54111 |

> **Note sur la courbe d'apprentissage :** La `val_loss` reste systématiquement inférieure à la `train_loss` — c'est l'effet normal du Dropout actif uniquement en mode entraînement. Il n'y a pas d'overfitting. Le scheduler LR descend en escalier (10⁻³ → 10⁻⁴ → 10⁻⁵), visible sur `ae_training_history.png`.

### Corrélations espace latent ↔ paramètres Gaia (Spearman)

| Paramètre Gaia | Axe latent 1 | Axe latent 2 |
|---|---|---|
| T_eff (K) | −0.279 | **−0.756** |
| log g | −0.236 | +0.107 |
| [Fe/H] | +0.267 | +0.494 |
| G_BP−G_RP | +0.268 | **+0.782** |

### 🔑 Résultat central NB03 : AE@2 ≈ PCA@10

| Méthode | Dimensions | MSE reconstruction |
|---|---|---|
| PCA | 1 | 0.8351 |
| PCA | 2 | 0.7317 |
| PCA | 5 | 0.6290 |
| PCA | 8 | 0.5722 |
| **Autoencodeur** | **2 ★** | **0.5411** |
| PCA | 10 | 0.5430 |
| PCA | 15 | 0.4803 |
| PCA | 20 | 0.4289 |
| PCA | 50 | 0.2443 |

> **En 2 dimensions latentes, l'autoencodeur reconstruit les spectres aussi bien qu'une PCA à 10 composantes.** La non-linéarité encode 5× plus d'information dans le même espace. Cela s'explique physiquement : la séquence stellaire est une **variété courbée** dans l'espace des features, pas un sous-espace euclidien plat.

### Détection d'anomalies — propriété émergente

L'autoencodeur a été entraîné **uniquement sur des étoiles**. Il repousse les objets non-stellaires vers des coordonnées latentes extrêmes et les reconstruit très mal :

| Classe | n | MSE médiane | MSE moyenne | Rapport vs étoiles |
|---|---|---|---|---|
| STAR | 42 862 | 0.281 | 0.531 | ×1 (référence) |
| GALAXY | 54 | 1.824 | 6.107 | **×6.5** |
| QSO | 4 | 30.062 | 28.884 | **×107** |

> Un seuil MSE > 2.0 suffit à détecter automatiquement les non-étoiles **sans supervision ni labels**. Sur LAMOST complet (~9M spectres), ce détecteur pourrait identifier les contaminations sans annotation manuelle.

### Interpolation latente — validation de la continuité

L'interpolation linéaire dans l'espace latent entre une étoile froide et une étoile chaude produit des reconstructions physiquement cohérentes :

- **Étoile froide :** T_eff = 4 200 K
- **Étoile chaude :** T_eff = 7 984 K
- **Distance euclidienne dans l'espace latent :** 12.84 unités
- **15 pas d'interpolation** → transitions spectrales continues et physiquement réalistes

---

## 📐 Synthèse comparative

| Méthode | ρ(axe 1, T_eff) | MSE (2 dims) | Paramétrique | Non-linéaire | Interprétable |
|---|---|---|---|---|---|
| **PCA** (PC1/PC2) | +0.832 | 0.7317 | ✅ Oui | ❌ Non | ✅ Oui (loadings) |
| **UMAP** | −0.333 | N/A | ⚠️ Partiel | ✅ Oui | ❌ Non |
| **Autoencodeur** (z=2) | −0.279 | **0.5411** | ✅ Oui | ✅ Oui | ⚠️ Partielle |

### Message central

> 🔵 **PCA** : la plus interprétable — les loadings correspondent directement à la physique (Hα vs Mg b).  
> 🟠 **UMAP** : la plus expressive — révèle la non-linéarité de la variété stellaire.  
> 🟢 **Autoencodeur** : le meilleur compromis reconstruction/dimensions, avec une détection d'anomalies émergente.

---

## 🖼️ Guide des figures

| Fichier | Notebook | Point clé |
|---|---|---|
| `pca_variance_explained.png` | NB01 | 91 PCs pour 95 % (features) vs 5 PCs (spectres bruts) |
| `pca_loadings_pc1.png` | NB01 | Hα vs Mg_b/CaII : axe température |
| `pca_loadings_pc2.png` | NB01 | FeH_proxy/metal_index : axe métallicité |
| `pca_eigenspectra.png` | NB01 | Artefact jonction bras LAMOST à 6 400 Å |
| `pca_correlation_heatmap.png` | NB01 | PC9↔RA, PC10↔Dec : contamination spatiale |
| `pca_scores_grid.png` | NB01 | Gradient T_eff le long de PC1 |
| `hr_diagram_pca_pc1.png` | NB01 | PCA retrouve la séquence HR sans labels |
| `umap_classes.png` | NB02 | Structure filamentaire vs blob PCA |
| `umap_negative_control.png` | NB02 | Validation : structures réelles vs données permutées |
| `stability_umap.png` | NB02 | Procrustes moy. 0.036 : très stable |
| `stability_tsne.png` | NB02 | Procrustes moy. 0.00025 : quasi-déterministe |
| `umap_sensitivity_n_neighbors.png` | NB02 | Robustesse aux hyperparamètres |
| `ae_training_history.png` | NB03 | Courbe saine, scheduler LR visible |
| `ae_latent_grid_zoomed.png` | NB03 | Gradient T_eff sur axe latent 2 |
| `ae_vs_pca_mse.png` | NB03 | **AE@2 ≈ PCA@10 : résultat central** |
| `ae_error_distribution_logscale.png` | NB03 | Séparation spectaculaire STAR/GALAXY/QSO |
| `ae_latent_interpolation.png` | NB03 | Continuité de l'espace latent |
| `ae_reconstruction_examples.png` | NB03 | Spectre original vs reconstruit |
| **`synthesis_pca_umap_ae.png`** | NB03 | **Figure centrale : 3 méthodes × 3 colorations** |

---

## 📚 Livrables académiques

### Symposium écrit — 24 avril 2026

**Structure du rapport :**

1. Résumé / Abstract (150–200 mots)
2. Introduction — contexte + question de recherche
3. Données et pipeline — LAMOST DR5 × Gaia DR3
4. Méthodes — PCA · UMAP · Autoencodeur
5. Résultats — *(cœur du rapport, 600–800 mots)*
6. Discussion — limites et interprétation
7. Conclusion — réponse à la question de recherche
8. Références

### Présentation orale — 30 avril / 1er mai 2026

**Lieu :** Musée de la Civilisation de Québec  
**Durée :** 12–15 minutes + questions  
**Arc narratif :**

| Acte | Message | Durée |
|---|---|---|
| 1 — Le problème | Des milliards d'étoiles, des millions de spectres : comment voir la structure cachée ? | 3–4 min |
| 2 — Les 3 regards | PCA, UMAP, autoencodeur : trois façons de comprimer l'espace spectral | 6–8 min |
| 3 — La surprise | L'AE fait aussi bien que la PCA à 10 dims avec seulement 2, et détecte les anomalies gratuitement | 3–4 min |

---

## 📖 Références

- Jolliffe, I.T. (2002). *Principal Component Analysis*, 2nd ed. Springer.
- McInnes, L., Healy, J., & Melville, J. (2018). [UMAP: Uniform Manifold Approximation and Projection](https://arxiv.org/abs/1802.03426). *arXiv:1802.03426*.
- Hinton, G.E. & Salakhutdinov, R.R. (2006). Reducing the Dimensionality of Data with Neural Networks. *Science*, 313, 504–507.
- Cui, X.-Q. et al. (2012). The Large Sky Area Multi-Object Fiber Spectroscopic Telescope (LAMOST). *Research in Astronomy and Astrophysics*, 12, 1197.
- Gaia Collaboration (2023). Gaia Data Release 3 summary. *A&A*, 674, A1.
- Portillo, S.K.N. et al. (2020). Improved Representation Learning for Stellar Spectra. *AJ*, 160, 45.

---

## ⚠️ Note aux collaborateurs

> Tous les chiffres, figures et conclusions de ce README proviennent des **notebooks exécutés par Alex** sur le jeu de données complet. Ne pas modifier les résultats numériques sans validation. Pour toute question sur les résultats, les notebooks ou les données, contacter Alex directement.

---

<div align="center">

**AstroSpectro · PHY-3500 · Université Laval · 2026**

*Ce projet fait partie de [AstroSpectro](https://github.com/PhD-Brown/AstroSpectro) — un pipeline de classification spectrale stellaire basé sur LAMOST DR5.*

</div>
