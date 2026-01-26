#!/usr/bin/env python3
"""
Script de vérification de la structure du dépôt PHY-3500
Exécutez ce script pour vérifier que tout est en place.
"""

import os
import sys
from pathlib import Path

def check_structure():
    """Vérifie que la structure du dépôt est correcte."""
    
    print("=" * 60)
    print("VÉRIFICATION DE LA STRUCTURE DU DÉPÔT PHY-3500")
    print("=" * 60)
    print()
    
    errors = []
    warnings = []
    
    # Vérifier les TPs
    tps = ["TP_1", "TP_2", "TP_3", "TP_4", "TP_final"]
    
    print("📁 Vérification des dossiers de TPs...")
    for tp in tps:
        if not os.path.isdir(tp):
            errors.append(f"Dossier manquant: {tp}")
        else:
            print(f"  ✓ {tp}")
            
            # Vérifier la structure interne
            subdirs = ["notebooks", "src", "data", "figs", "results"]
            for subdir in subdirs:
                path = os.path.join(tp, subdir)
                if not os.path.isdir(path):
                    errors.append(f"Sous-dossier manquant: {path}")
                    
            # Vérifier les notebooks
            notebooks = [
                f"{tp}/notebooks/draft_m1.ipynb",
                f"{tp}/notebooks/draft_m2.ipynb",
                f"{tp}/notebooks/draft_m3.ipynb",
            ]
            
            if tp == "TP_final":
                notebooks.append(f"{tp}/notebooks/TPfinal_nom1_nom2_nom3.ipynb")
            else:
                tp_num = tp.split("_")[1]
                notebooks.append(f"{tp}/notebooks/TP{tp_num}_nom1_nom2_nom3.ipynb")
                
            for nb in notebooks:
                if not os.path.isfile(nb):
                    errors.append(f"Notebook manquant: {nb}")
                    
            # Vérifier les modules Python
            py_files = [
                f"{tp}/src/__init__.py",
                f"{tp}/src/utils.py",
                f"{tp}/src/analysis.py",
            ]
            for py_file in py_files:
                if not os.path.isfile(py_file):
                    errors.append(f"Module Python manquant: {py_file}")
    
    print()
    print("📄 Vérification des fichiers de configuration...")
    
    config_files = {
        ".gitignore": "Configuration Git",
        ".gitattributes": "Configuration nbdime",
        ".pre-commit-config.yaml": "Configuration pre-commit",
        "requirements.txt": "Dépendances Python",
        "setup_venv.sh": "Script setup Linux/Mac",
        "setup_venv.ps1": "Script setup Windows",
        "README.md": "Documentation principale",
        "CONTRIBUTING.md": "Guide de contribution",
    }
    
    for file, desc in config_files.items():
        if os.path.isfile(file):
            print(f"  ✓ {file} ({desc})")
        else:
            errors.append(f"Fichier manquant: {file}")
    
    print()
    print("🔍 Vérification du contenu des fichiers...")
    
    # Vérifier que requirements.txt contient les packages essentiels
    if os.path.isfile("requirements.txt"):
        with open("requirements.txt", "r") as f:
            content = f.read()
            essential_packages = ["numpy", "matplotlib", "jupyter", "nbstripout", "black", "isort", "pre-commit", "nbdime"]
            for pkg in essential_packages:
                if pkg.lower() in content.lower():
                    print(f"  ✓ Package {pkg} présent dans requirements.txt")
                else:
                    warnings.append(f"Package {pkg} absent de requirements.txt")
    
    # Vérifier que les scripts sont exécutables (Linux uniquement)
    if os.name == 'posix':
        if os.path.isfile("setup_venv.sh"):
            if os.access("setup_venv.sh", os.X_OK):
                print(f"  ✓ setup_venv.sh est exécutable")
            else:
                warnings.append("setup_venv.sh n'est pas exécutable (chmod +x setup_venv.sh)")
    
    print()
    print("=" * 60)
    print("RÉSULTATS")
    print("=" * 60)
    
    if not errors and not warnings:
        print()
        print("✅ SUCCÈS! La structure du dépôt est complète et correcte.")
        print()
        print("Prochaines étapes:")
        print("  1. Exécutez ./setup_venv.sh (ou .\\setup_venv.ps1 sur Windows)")
        print("  2. Activez l'environnement: source venv/bin/activate")
        print("  3. Lancez Jupyter: jupyter notebook")
        print()
        return 0
    else:
        if errors:
            print()
            print(f"❌ {len(errors)} ERREUR(S) DÉTECTÉE(S):")
            for error in errors:
                print(f"  - {error}")
        
        if warnings:
            print()
            print(f"⚠️  {len(warnings)} AVERTISSEMENT(S):")
            for warning in warnings:
                print(f"  - {warning}")
        
        print()
        return 1

if __name__ == "__main__":
    sys.exit(check_structure())
