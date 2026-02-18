#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TP reconstruction TDM (CT)
# Prof: Philippe Després
# programme: Dmitri Matenine (dmitri.matenine.1@ulaval.ca)


# fichier contenant la description de la géométrie
# d'acquisition
# et de reconstruction

import numpy as np

### VARIABLES ###

### paramètres d'acquisition ###

## largeur d'un élément de détecteur (cm)
pixsize = 0.165

## taille du détecteur (nombre d'échantillons)
nbpix = 336

### paramètres de reconstruction ###

## taille de la grille d'image (carrée)
nbvox = 96 # options: 96, 192

## taille du voxel (carré) (cm)
voxsize = 0.4 # option: 0.4, 0.2

## fichiers d'entrée
dataDir = "./data/"
anglesFile = "angles.txt"
sinogramFile = "sinogram-password.txt"


### FONCTIONS UTILITAIRES ###

def get_proj_value_nearest(sinogram_row, t_pixel):
    """
    Retourne la valeur de projection par plus proche voisin.

    Paramètres
    ----------
    sinogram_row : ndarray (1D)
        Une projection (ligne du sinogramme).
    t_pixel : float
        Position fractionnaire sur le détecteur (pixels).

    Retour
    ------
    float
        Valeur d'atténuation au pixel le plus proche.
    """
    nbpix = len(sinogram_row)
    idx = int(np.round(t_pixel))
    if 0 <= idx < nbpix:
        return sinogram_row[idx]
    return 0.0


def get_proj_value_linear(sinogram_row, t_pixel):
    """
    Retourne la valeur de projection par interpolation linéaire entre
    les deux pixels adjacents du détecteur. (Q5)

    Au lieu d'arrondir au plus proche voisin, on interpole :
        valeur = (1 - frac) * sinogram_row[i0] + frac * sinogram_row[i1]
    où i0 = floor(t_pixel), i1 = i0 + 1 et frac = t_pixel - i0.

    Paramètres
    ----------
    sinogram_row : ndarray (1D)
        Une projection (ligne du sinogramme).
    t_pixel : float
        Position fractionnaire sur le détecteur (pixels).

    Retour
    ------
    float
        Valeur d'atténuation interpolée linéairement.
    """
    nbpix = len(sinogram_row)
    i0 = int(np.floor(t_pixel))
    i1 = i0 + 1
    frac = t_pixel - i0

    if 0 <= i0 < nbpix and 0 <= i1 < nbpix:
        return (1.0 - frac) * sinogram_row[i0] + frac * sinogram_row[i1]
    elif 0 <= i0 < nbpix:
        return sinogram_row[i0]
    elif 0 <= i1 < nbpix:
        return sinogram_row[i1]
    return 0.0

