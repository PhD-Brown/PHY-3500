# Script de configuration de l'environnement virtuel Python pour PHY-3500
# Usage: .\setup_venv.ps1

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Configuration de l'environnement virtuel" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Vérifier si Python 3 est installé
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Version de Python détectée:" -ForegroundColor Green
    Write-Host $pythonVersion
} catch {
    Write-Host "Erreur: Python 3 n'est pas installé." -ForegroundColor Red
    Write-Host "Veuillez installer Python 3.8 ou supérieur depuis python.org" -ForegroundColor Yellow
    exit 1
}

# Créer l'environnement virtuel
Write-Host ""
Write-Host "Création de l'environnement virtuel..." -ForegroundColor Yellow
python -m venv venv

# Activer l'environnement virtuel
Write-Host "Activation de l'environnement virtuel..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Mettre à jour pip
Write-Host ""
Write-Host "Mise à jour de pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Installer les dépendances
Write-Host ""
Write-Host "Installation des dépendances..." -ForegroundColor Yellow
pip install -r requirements.txt

# Installer les hooks pre-commit
Write-Host ""
Write-Host "Installation des hooks pre-commit..." -ForegroundColor Yellow
pre-commit install

# Configuration de nbdime
Write-Host ""
Write-Host "Configuration de nbdime pour git..." -ForegroundColor Yellow
nbdime config-git --enable

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Installation terminée avec succès!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Pour activer l'environnement virtuel:" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Pour démarrer Jupyter:" -ForegroundColor Yellow
Write-Host "  jupyter notebook"
Write-Host ""
Write-Host "Pour désactiver l'environnement virtuel:" -ForegroundColor Yellow
Write-Host "  deactivate"
Write-Host ""
