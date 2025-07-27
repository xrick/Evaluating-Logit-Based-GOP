# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project implementing Goodness of Pronunciation (GOP) scoring for mispronunciation detection using CTC-based models. The project specifically implements the "Phoneme-Perturbed Alignment-Free GOP (PP-AF GOP)" method from the paper "Enhancing GOP in CTC-Based Mispronunciation Detection with Phonological Knowledge."

## Core Architecture

### Main Components

- **main.py**: Entry point that orchestrates the entire experiment workflow
- **gop_calculators.py**: Core GOP calculation algorithms, including PP-AF GOP implementation
- **phoneme_map.py**: Phoneme confusion map for Restricted Phoneme Substitutions (RPS) mode
- **data_loader.py**: Mock data generator for testing and demonstration
- **evaluation.py**: Performance metrics calculation utilities

### Key Concepts

1. **PP-AF GOP (Phoneme-Perturbed Alignment-Free GOP)**: Main algorithm that calculates GOP scores by comparing original CTC loss with perturbed sequence losses
2. **RPS Mode**: Uses predefined phoneme confusion map for realistic substitutions
3. **UPS Mode**: Uses unrestricted phoneme substitutions from the entire vocabulary
4. **CTC Loss**: Uses Wav2Vec2 model to compute Connectionist Temporal Classification loss

## Setup and Dependencies

Install dependencies with:
```bash
pip install -r requirements.txt
```

Key dependencies:
- torch (PyTorch for neural networks)
- transformers (Hugging Face models)
- phonemizer (text-to-phoneme conversion)
- librosa (audio processing)
- scikit-learn (evaluation metrics)

## Running the Code

Execute the main experiment:
```bash
python main.py
```

This will:
1. Load the pretrained Wav2Vec2 model (`facebook/wav2vec2-xisr-53-espeak-cv-ft`)
2. Generate mock audio samples for testing
3. Calculate GOP scores using both RPS and UPS modes
4. Display comparative results

## Development Notes

### Audio Processing
- Audio files are expected at 16kHz sample rate
- Uses librosa for audio loading and processing
- Mock data generates synthetic sine waves for testing

### Model Integration
- Uses Wav2Vec2ForCTC from Hugging Face transformers
- Model processes audio to logits for CTC loss calculation
- Supports GPU acceleration when available

### Phoneme Processing
- Uses espeak backend for phonemization
- Implements basic phoneme cleaning (removes stress markers)
- Handles phoneme-to-token conversion for the model vocabulary

### Error Handling
- Graceful handling of infinite CTC losses
- Validation of phonemes against model vocabulary
- Comprehensive exception handling in main workflow

## Key Functions

### gop_calculators.py:line_numbers
- `init_model_and_processor()`: Loads Wav2Vec2 model and processor
- `get_ctc_loss()`: Computes CTC loss for given logits and labels
- `calculate_pp_af_gop()`: Core PP-AF GOP implementation with perturbation logic

### phoneme_map.py:line_numbers
- `get_restricted_substitutions()`: Returns allowed phoneme substitutions for RPS mode
- `PHONEME_CONFUSION_MAP`: Predefined confusion patterns based on phonetic similarity

## Testing and Validation

The project uses mock data for demonstration. To use with real data:
1. Modify `data_loader.py` to load actual audio files and transcripts
2. Ensure audio files are in WAV format at 16kHz
3. Provide canonical transcripts for GOP comparison

## Research Context

This implementation follows the methodology described in the research paper, specifically:
- Section 2.3.3 for PP-AF GOP algorithm
- Equation (4) for GOP score calculation: `GOP = min(L_perturbed) - L_original`
- Phoneme confusion patterns from Section 2.3.1

Higher GOP scores indicate better pronunciation quality; negative scores may indicate mispronunciation.