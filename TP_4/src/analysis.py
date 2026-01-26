"""
Fonctions d'analyse pour le TP 4
"""

import numpy as np
import matplotlib.pyplot as plt


def analyze_data(data):
    """
    Analyse des données.
    
    Parameters
    ----------
    data : array-like
        Données à analyser
        
    Returns
    -------
    dict
        Dictionnaire contenant les résultats de l'analyse
        
    TODO
    ----
    - Implémentez vos fonctions d'analyse ici
    - Retournez des résultats structurés
    """
    # TODO: Implémentez votre analyse ici
    results = {
        'mean': np.mean(data),
        'std': np.std(data),
    }
    return results


def plot_results(data, **kwargs):
    """
    Visualisation des résultats.
    
    Parameters
    ----------
    data : array-like
        Données à visualiser
    **kwargs : dict
        Arguments supplémentaires pour la visualisation
        
    Returns
    -------
    fig, ax
        Figure et axes matplotlib
        
    TODO
    ----
    - Implémentez vos fonctions de visualisation ici
    - Sauvegardez les figures dans le dossier figs/
    """
    # TODO: Implémentez votre visualisation ici
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set_xlabel('Index')
    ax.set_ylabel('Valeur')
    return fig, ax


# TODO: Ajoutez vos fonctions d'analyse ici
