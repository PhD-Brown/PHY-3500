#!/bin/bash
# Script de configuration de l'environnement virtuel Python pour PHY-3500
# Usage: ./setup_venv.sh

echo "=========================================="
echo "Configuration de l'environnement virtuel"
echo "=========================================="

# Vérifier si Python 3 est installé
if ! command -v python3 &> /dev/null; then
    echo "Erreur: Python 3 n'est pas installé."
    echo "Veuillez installer Python 3.8 ou supérieur."
    exit 1
fi

# Afficher la version de Python
echo "Version de Python détectée:"
python3 --version

# Créer l'environnement virtuel
echo ""
echo "Création de l'environnement virtuel..."
python3 -m venv venv

# Activer l'environnement virtuel
echo "Activation de l'environnement virtuel..."
source venv/bin/activate

# Mettre à jour pip
echo ""
echo "Mise à jour de pip..."
pip install --upgrade pip

# Installer les dépendances
echo ""
echo "Installation des dépendances..."
pip install -r requirements.txt

# Installer les hooks pre-commit
echo ""
echo "Installation des hooks pre-commit..."
pre-commit install

# Configuration de nbdime
echo ""
echo "Configuration de nbdime pour git..."
nbdime config-git --enable

echo ""
echo "=========================================="
echo "Installation terminée avec succès!"
echo "=========================================="
echo ""
echo "Pour activer l'environnement virtuel:"
echo "  source venv/bin/activate"
echo ""
echo "Pour démarrer Jupyter:"
echo "  jupyter notebook"
echo ""
echo "Pour désactiver l'environnement virtuel:"
echo "  deactivate"
echo ""
