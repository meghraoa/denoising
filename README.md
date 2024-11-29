# Denoising Audio Project 🎧

## Introduction
Ce projet vise à débruiter des enregistrements de voix en estimant le signal vocal d'origine à partir de signaux bruités contenant des bruits d'ambiance de rue. L'objectif est de restaurer une voix claire à partir de données avec un **SNR (Signal to Noise Ratio)** compris entre 0 et 20 dB.

## Structure des données
Les données sont organisées comme suit :  
- **Train** :  
  - `audio/voice_origin/train` : Enregistrements de voix sans bruit (voix d'origine).  
  - `audio/denoising/train` : Enregistrements de voix bruités (ambiance de rue).  
  - Les fichiers des deux dossiers correspondent par leur nom.  

- **Test** :  
  - `audio/voice_origin/test` : Enregistrements propres (ensemble de test).  
  - `audio/denoising/test` : Enregistrements bruités (ensemble de test).  

- **Ensembles réduits** :  
  - `audio/voice_origin/train_small` et `audio/denoising/train_small` : Sous-ensemble pour effectuer des essais rapides.

## Objectif
Estimer le signal vocal propre à partir d’un signal bruité tout en optimisant les métriques suivantes :  
- **PESQ (Perceptual Evaluation of Speech Quality)** : Évalue la qualité perceptuelle de la voix estimée.  
- **STOI (Short-Time Objective Intelligibility)** : Évalue l’intelligibilité de la voix.

## Installation
1. Clonez ce dépôt :  
   ```bash
   git clone <url_du_repo>
   cd denoising
   ```
