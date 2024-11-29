# Denoising Audio Project 🎧

## Introduction
This project aims to denoise voice recordings by estimating the original speech signal from noisy audio containing street ambient noise. The goal is to restore clear speech from data with an **SNR (Signal-to-Noise Ratio)** between 0 and 20 dB.

## Data Structure
The data is organized as follows:  
- **Train**:  
  - `audio/voice_origin/train`: Clean voice recordings (original speech).  
  - `audio/denoising/train`: Noisy voice recordings (street ambiance).  
  - Files in both directories correspond by name.  

- **Test**:  
  - `audio/voice_origin/test`: Clean recordings (test set).  
  - `audio/denoising/test`: Noisy recordings (test set).  

- **Reduced Sets**:  
  - `audio/voice_origin/train_small` and `audio/denoising/train_small`: Subset for quick testing.

## Objective
Estimate the clean speech signal from noisy audio while optimizing the following metrics:  
- **PESQ (Perceptual Evaluation of Speech Quality)**: Evaluates the perceptual quality of the estimated speech.  
- **STOI (Short-Time Objective Intelligibility)**: Assesses the intelligibility of the estimated speech.

## Installation
1. Clone this repository:  
   ```bash
   git clone <repo_url>
   cd denoising
   ```
