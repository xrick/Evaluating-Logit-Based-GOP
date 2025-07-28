# GEMINI.md

This file provides guidance to Google's Gemini models when working with code in this repository.

## Project Overview

This project is a Python implementation for evaluating "Goodness of Pronunciation" (GOP). It specifically implements the **Phoneme-Perturbed Alignment-Free GOP (PP-AF GOP)** method, which uses a CTC-based acoustic model (Wav2Vec2) to score pronunciation quality without needing forced alignment. The goal is to detect mispronunciations by analyzing the acoustic model's behavior when presented with phonetically perturbed versions of a transcript.

The implementation is based on the research paper "Enhancing GOP in CTC-Based Mispronunciation Detection with Phonological Knowledge."

## Core Architecture

The project is structured into several key Python modules:

-   `main.py`: The main entry point for running the GOP evaluation experiment. It orchestrates data loading, model initialization, and GOP calculation.
-   `gop_calculators.py`: Contains the core logic for the PP-AF GOP algorithm. This is where the CTC loss is calculated for original and perturbed phoneme sequences.
-   `data_loader.py`: Provides functionality to load data. The current version generates mock audio and transcript data for demonstration purposes.
-   `phoneme_map.py`: Defines a phoneme confusion map used for the "Restricted Phoneme Substitutions" (RPS) mode, which simulates common, linguistically plausible pronunciation errors.
-   `evaluation.py`: Includes utility functions to calculate standard classification metrics (Accuracy, F1-score, etc.) to evaluate the performance of the GOP-based mispronunciation detection.
-   `requirements.txt`: Lists all the necessary Python dependencies.

## Key Concepts

-   **Goodness of Pronunciation (GOP)**: A score that quantifies how well a spoken phoneme matches its canonical form.
-   **PP-AF GOP**: An alignment-free method that calculates GOP by comparing the CTC loss of the original phoneme sequence (`L_original`) with the minimum loss from a set of perturbed sequences (`L_perturbed`). The GOP score is `min(L_perturbed) - L_original`.
-   **CTC (Connectionist Temporal Classification)**: A loss function used for training sequence-to-sequence models like Wav2Vec2, which allows the model to learn alignments between input (audio frames) and output (phonemes) on its own.
-   **RPS (Restricted Phoneme Substitutions)**: A GOP calculation mode that only uses a predefined, linguistically-informed set of phoneme substitutions for perturbations. This is defined in `phoneme_map.py`.
-   **UPS (Unrestricted Phoneme Substitutions)**: A GOP calculation mode that perturbs a phoneme by substituting it with any other phoneme from the model's vocabulary.

## Setup and How to Run

### 1. Install Dependencies

First, ensure all required Python packages are installed.

```bash
pip install -r requirements.txt
```

You also need to install the `espeak` backend for the `phonemizer` library:
*   **Ubuntu/Debian**: `sudo apt-get install espeak espeak-data`
*   **macOS**: `brew install espeak`

### 2. Run the Main Script

To run the demonstration with mock data, execute the main script:

```bash
python main.py
```

This command will:
1.  Initialize the `facebook/wav2vec2-xisr-53-espeak-cv-ft` model from Hugging Face.
2.  Generate mock audio samples in a `.mock_data` directory.
3.  Process each sample, calculating GOP scores using both RPS and UPS modes.
4.  Print the phonemes, their corresponding GOP scores, and the average GOP for each audio sample.

## Development and Modification Guide

### Using Custom Data

To use your own audio files and transcripts:
1.  Modify `data_loader.py` to load your data. Your audio should be 16kHz WAV files. You'll need a function that returns a list of dictionaries, similar to `load_mock_data`.
2.  In `main.py`, replace the call to `load_mock_data()` with your new data loading function.

### Modifying Phoneme Substitutions

The phoneme confusion map for RPS mode can be customized by editing the `PHONEME_CONFUSION_MAP` dictionary in `phoneme_map.py`. This is useful for adapting the system to specific learner language backgrounds (L1s).

### Core Algorithm

The main algorithm is in `calculate_pp_af_gop` within `gop_calculators.py`. This is the function to modify if you want to experiment with different perturbation strategies or GOP calculation formulas. The key steps are:
1.  Get logits from the Wav2Vec2 model for an audio input.
2.  Phonemize the canonical transcript.
3.  Calculate `L_original` (the CTC loss for the correct phoneme sequence).
4.  For each phoneme, create perturbed sequences (by deletion and substitution).
5.  Calculate the CTC loss for each perturbed sequence to find `min(L_perturbed)`.
6.  Compute the final GOP score.
