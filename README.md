# Source Separation 🎧

The goal of this project is to jointly estimate the voice component and the noise component of an audio recording. For this project, you have:

## Training Data
- A folder containing numbered subfolders (e.g., 0001 or 1256).
- Inside each subfolder, you will find three `.wav` files: `mix_snr_XX.wav`, `voice.wav`, and `noise.wav`.
- `voice.wav` and `noise.wav` are the ground truth files to estimate. `mix_snr_XX.wav` is the mixture of both sources, with an SNR of XX for the voice component (and -XX for the noise component).

The test set is organized in the same way.
