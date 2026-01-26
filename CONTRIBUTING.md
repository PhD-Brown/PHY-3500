# Guide de contribution - PHY-3500

Ce document fournit des conseils pour bien collaborer sur les TPs.

## 🎯 Principes de base

### 1. Communication avant tout
- Discutez avec votre équipe avant de commencer à travailler
- Utilisez les issues GitHub pour suivre les tâches
- Commentez vos Pull Requests de manière claire

### 2. Travail sur les notebooks

#### ✅ À FAIRE
- Utilisez votre brouillon personnel (`draft_m*.ipynb`) pour vos expérimentations
- Coordonnez-vous avant de modifier le notebook principal
- Testez que toutes les cellules s'exécutent avant de committer
- Écrivez des messages de commit descriptifs
- Faites des commits petits et fréquents

#### ❌ À ÉVITER
- Ne travaillez jamais à plusieurs simultanément sur le même notebook
- N'oubliez pas de synchroniser (`git pull`) avant de commencer
- Ne commitez pas de notebooks avec des erreurs
- N'ignorez pas les conflits - résolvez-les immédiatement
- Ne poussez pas de gros fichiers de données (utilisez .gitignore)

### 3. Organisation du code

#### Structure recommandée
```python
# 1. Imports
import numpy as np
import matplotlib.pyplot as plt
from src.utils import ma_fonction

# 2. Configuration
plt.rcParams['figure.figsize'] = (10, 6)

# 3. Fonctions locales (si nécessaire)
def fonction_specifique():
    pass

# 4. Code principal
# Vos calculs ici

# 5. Visualisation et sauvegarde
fig, ax = plt.subplots()
# ...
fig.savefig('../figs/mon_graphique.png', dpi=300, bbox_inches='tight')
```

#### Bonnes pratiques Python
- Utilisez des noms de variables descriptifs en français
- Commentez les parties complexes
- Documentez vos fonctions avec des docstrings
- Évitez les nombres magiques (utilisez des constantes nommées)
- Respectez PEP 8 (automatique avec black)

### 4. Gestion des modules

#### Dans `src/utils.py`
Placez les fonctions utilitaires génériques :
- Lecture/écriture de fichiers
- Conversions d'unités
- Fonctions mathématiques réutilisables

#### Dans `src/analysis.py`
Placez les fonctions d'analyse spécifiques :
- Calculs physiques
- Analyse statistique
- Visualisations complexes

#### Exemple de module bien documenté
```python
"""
Module d'analyse pour TP_1
"""

import numpy as np


def calculer_energie(masse, vitesse):
    """
    Calcule l'énergie cinétique.

    Parameters
    ----------
    masse : float
        Masse en kg
    vitesse : float
        Vitesse en m/s

    Returns
    -------
    float
        Énergie cinétique en Joules

    Examples
    --------
    >>> calculer_energie(1.0, 10.0)
    50.0
    """
    return 0.5 * masse * vitesse**2
```

### 5. Workflow Git recommandé

#### Pour une nouvelle section du TP
```bash
# 1. Créer une branche
git checkout -b tp1-section2

# 2. Travailler sur votre branche
# ... modifications ...

# 3. Tester localement
jupyter nbconvert --to notebook --execute --inplace votre_notebook.ipynb

# 4. Committer
git add .
git commit -m "TP1: Ajouter analyse de la section 2"

# 5. Pousser
git push origin tp1-section2

# 6. Créer une Pull Request sur GitHub
# 7. Demander une révision
# 8. Fusionner après approbation
```

#### Messages de commit

**Format recommandé:**
```
TPx: Description courte du changement

- Détail 1
- Détail 2
```

**Exemples:**
```
TP1: Ajouter import des données expérimentales
TP2: Corriger calcul de l'erreur relative
TP3: Améliorer visualisation des résultats
```

### 6. Révision de code

#### Ce qu'il faut vérifier
- [ ] Le code s'exécute sans erreur
- [ ] Les résultats semblent corrects
- [ ] Le code est commenté et lisible
- [ ] Les graphiques ont des légendes et des labels
- [ ] Les unités sont correctes
- [ ] Pas de code dupliqué
- [ ] Les fichiers sont organisés correctement

#### Commentaires constructifs
**Bon ✅**
> "Cette approche fonctionne bien. Pour améliorer la lisibilité, on pourrait extraire cette logique dans une fonction séparée."

**À éviter ❌**
> "Ce code est mauvais."

### 7. Gestion des conflits

#### Prévention
- Communiquez avec votre équipe
- Travaillez sur des sections différentes
- Synchronisez souvent (`git pull`)

#### Résolution
Si un conflit survient sur un notebook:

```bash
# Option 1: Utiliser nbdime
nbdime mergetool

# Option 2: Choisir une version
git checkout --ours notebook.ipynb   # Garder votre version
git checkout --theirs notebook.ipynb # Garder leur version
git add notebook.ipynb
git commit
```

### 8. Fichiers et données

#### Organisation
```
TP_1/
├── data/
│   ├── raw/              # Données brutes (ne pas modifier)
│   └── processed/        # Données traitées
├── figs/
│   ├── exploration/      # Graphiques exploratoires
│   └── final/            # Graphiques pour le rapport
└── results/
    ├── numerical/        # Résultats numériques
    └── stats/            # Statistiques
```

#### Nommage
- Utilisez des noms descriptifs
- Incluez la date si pertinent: `resultats_2026-01-26.csv`
- Pas d'espaces, utilisez `_` ou `-`

#### .gitignore
Ajoutez les gros fichiers à `.gitignore`:
```
# Données volumineuses
*.hdf5
*.h5
data/raw/*.csv

# Fichiers temporaires
*_temp.csv
*_backup.ipynb
```

### 9. Documentation

#### Dans le notebook
- Titre clair pour chaque section
- Explication de la méthode avant le code
- Interprétation des résultats après
- Conclusion à la fin

#### Cellules Markdown
Utilisez Markdown pour structurer:

```markdown
## Section 1: Analyse préliminaire

### Objectif
Analyser la distribution des données...

### Méthode
Nous utilisons une régression linéaire parce que...

### Résultats
Les résultats montrent que...
```

### 10. Checklist finale avant remise

- [ ] Toutes les cellules s'exécutent dans l'ordre (Kernel > Restart & Run All)
- [ ] Les informations de l'équipe sont remplies
- [ ] Le fichier est renommé correctement
- [ ] Les figures sont dans `figs/`
- [ ] Les résultats sont dans `results/`
- [ ] Le code est propre et commenté
- [ ] Les graphiques ont des titres, légendes et labels avec unités
- [ ] Les conclusions sont présentes
- [ ] Pas d'erreurs ou de warnings
- [ ] Les outputs des notebooks sont nettoyés (fait automatiquement par pre-commit)
- [ ] Tous les changements sont committés et poussés
- [ ] Les coéquipiers ont révisé

## 📞 Besoin d'aide?

- Consultez le [README.md](README.md)
- Créez une issue sur GitHub
- Demandez à vos coéquipiers
- Consultez la [documentation Jupyter](https://jupyter.org/documentation)

---

**Bonne collaboration! 🚀**
