# PHY-3500 - Physique Numérique

Dépôt Git pour les travaux pratiques en équipe du cours de Physique Numérique - H26

## 📁 Structure du dépôt

```
PHY-3500/
├── TP_1/
│   ├── notebooks/
│   │   ├── draft_m1.ipynb          # Brouillon membre 1
│   │   ├── draft_m2.ipynb          # Brouillon membre 2
│   │   ├── draft_m3.ipynb          # Brouillon membre 3
│   │   └── TP1_nom1_nom2_nom3.ipynb  # Template à renommer et remettre
│   ├── src/
│   │   ├── __init__.py
│   │   ├── utils.py                # Fonctions utilitaires
│   │   └── analysis.py             # Fonctions d'analyse
│   ├── data/                       # Données du TP
│   ├── figs/                       # Figures générées
│   └── results/                    # Résultats sauvegardés
├── TP_2/                           # Même structure
├── TP_3/                           # Même structure
├── TP_4/                           # Même structure
├── TP_final/                       # Même structure
├── .gitignore
├── .gitattributes
├── .pre-commit-config.yaml
├── requirements.txt
├── setup_venv.sh                   # Script Linux/Mac
├── setup_venv.ps1                  # Script Windows
└── README.md
```

## 🚀 Installation et configuration

### Prérequis
- Python 3.8 ou supérieur
- Git installé et configuré
- Compte GitHub avec accès au dépôt

### Installation

#### Sur Linux/Mac
```bash
# Cloner le dépôt
git clone https://github.com/PhD-Brown/PHY-3500.git
cd PHY-3500

# Exécuter le script de configuration
chmod +x setup_venv.sh
./setup_venv.sh

# Activer l'environnement virtuel
source venv/bin/activate
```

#### Sur Windows (PowerShell)
```powershell
# Cloner le dépôt
git clone https://github.com/PhD-Brown/PHY-3500.git
cd PHY-3500

# Permettre l'exécution de scripts (si nécessaire)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Exécuter le script de configuration
.\setup_venv.ps1

# Activer l'environnement virtuel
.\venv\Scripts\Activate.ps1
```

## 📓 Utilisation des notebooks

### Démarrer Jupyter
```bash
# Avec l'environnement virtuel activé
jupyter notebook
# ou
jupyter lab
```

### Workflow de travail

1. **Travail individuel sur les brouillons**
   - Chaque membre utilise son fichier `draft_m*.ipynb`
   - Ces fichiers ne sont PAS évalués
   - Utilisez-les pour vos essais et expérimentations

2. **Travail d'équipe sur le notebook final**
   - Renommez le template : `TP1_nom1_nom2_nom3.ipynb`
   - Complétez les sections ensemble
   - Assurez-vous que toutes les cellules s'exécutent sans erreur

3. **Utilisation des modules Python**
   - Placez vos fonctions réutilisables dans `src/`
   - Importez-les dans vos notebooks : `sys.path.append('../src')`
   - Documentez vos fonctions avec des docstrings

## 🔄 Workflow Git et collaboration

### Règles de base pour éviter les conflits

#### ⚠️ **IMPORTANT : Règles anti-conflits notebooks**

Les notebooks Jupyter peuvent causer des conflits Git difficiles à résoudre. Suivez ces règles :

1. **UN membre à la fois travaille sur le notebook final**
   - Coordonnez-vous avant de modifier le notebook principal
   - Utilisez les brouillons individuels pour le travail en parallèle

2. **Toujours synchroniser AVANT de travailler**
   ```bash
   git pull origin main
   ```

3. **Committez régulièrement et fréquemment**
   ```bash
   git add .
   git commit -m "Description claire des changements"
   git push origin main
   ```

4. **Nettoyez les outputs avant de committer**
   - Les hooks pre-commit le font automatiquement
   - Ou manuellement : `jupyter nbconvert --clear-output --inplace votre_notebook.ipynb`

### Workflow avec branches (recommandé)

Pour un travail plus sûr, utilisez des branches :

```bash
# Créer une branche pour une nouvelle fonctionnalité
git checkout -b tp1-analyse-donnees

# Travailler sur votre branche
# ... modifications ...

# Committer vos changements
git add .
git commit -m "Ajout de l'analyse des données pour TP1"

# Pousser votre branche
git push origin tp1-analyse-donnees

# Créer une Pull Request sur GitHub
# Faites réviser par vos coéquipiers
# Fusionnez dans main après approbation
```

### Workflow avec Pull Requests

1. **Créer une branche** pour chaque nouvelle fonctionnalité ou section
2. **Pousser la branche** sur GitHub
3. **Ouvrir une Pull Request (PR)**
4. **Révision par les coéquipiers**
   - Au moins un autre membre doit réviser
   - Vérifier que le code s'exécute
   - Vérifier la clarté et la documentation
5. **Fusionner** après approbation
6. **Supprimer la branche** après fusion

### En cas de conflit

Si vous rencontrez un conflit sur un notebook :

```bash
# Option 1 : Utiliser nbdime (recommandé)
nbdime mergetool

# Option 2 : Accepter une version
git checkout --theirs notebook.ipynb  # Garder la version distante
# ou
git checkout --ours notebook.ipynb    # Garder votre version

# Option 3 : Recommencer
# Sauvegarder votre travail ailleurs
git checkout main
git pull origin main
# Refaire vos modifications
```

## 🛠️ Outils de développement

### Pre-commit hooks

Les hooks automatiques s'exécutent avant chaque commit :
- `nbstripout` : Nettoie les outputs des notebooks
- `black` : Formate le code Python
- `isort` : Trie les imports
- Vérifications de base (trailing whitespace, etc.)

Pour exécuter manuellement :
```bash
pre-commit run --all-files
```

### nbdime - Diff pour notebooks

Comparer des notebooks :
```bash
nbdiff notebook1.ipynb notebook2.ipynb
```

Fusionner des notebooks :
```bash
nbmerge base.ipynb local.ipynb remote.ipynb
```

Interface web pour les diffs :
```bash
nbdiff-web notebook1.ipynb notebook2.ipynb
```

## 📋 Checklist avant de remettre un TP

- [ ] Le notebook est renommé avec les noms de l'équipe
- [ ] Toutes les cellules s'exécutent sans erreur (Kernel > Restart & Run All)
- [ ] Les informations de l'équipe sont remplies
- [ ] Les figures sont sauvegardées dans `figs/`
- [ ] Les résultats sont sauvegardés dans `results/`
- [ ] Le code est commenté et documenté
- [ ] Les outputs des notebooks sont nettoyés (fait automatiquement par pre-commit)
- [ ] Tous les changements sont committés et poussés
- [ ] Les coéquipiers ont révisé le travail

## 🆘 Aide et support

### Problèmes courants

**L'environnement virtuel ne s'active pas**
- Sur Windows : vérifiez la politique d'exécution des scripts
- Sur Linux/Mac : assurez-vous que le script est exécutable (`chmod +x setup_venv.sh`)

**Conflits Git dans les notebooks**
- Suivez les règles anti-conflits ci-dessus
- Utilisez nbdime pour résoudre les conflits
- En dernier recours, choisissez une version et refaites les modifications

**Les hooks pre-commit échouent**
- Lisez le message d'erreur
- Corrigez les problèmes signalés
- Les fichiers sont automatiquement modifiés par black et isort
- Re-commitez après les modifications automatiques

**Packages manquants**
```bash
pip install -r requirements.txt
```

## 📚 Ressources

- [Documentation Jupyter](https://jupyter.org/documentation)
- [Guide Git](https://git-scm.com/doc)
- [Documentation nbdime](https://nbdime.readthedocs.io/)
- [Documentation pre-commit](https://pre-commit.com/)
- [Style guide Python (PEP 8)](https://pep8.org/)

## 👥 Équipe

Ce dépôt est conçu pour le travail en équipe de 3 personnes. Respectez vos coéquipiers :
- Communiquez avant de travailler sur le notebook principal
- Committez régulièrement avec des messages clairs
- Révisez le travail des autres
- Aidez-vous mutuellement

---

**Bon travail et bonne collaboration ! 🎓**
