# Evaluating Logit-Based GOP Scores for Mispronunciation Detection

A Python implementation of Goodness of Pronunciation (GOP) scoring for automatic mispronunciation detection using CTC-based neural networks. This project implements the **Phoneme-Perturbed Alignment-Free GOP (PP-AF GOP)** method described in the research paper "Enhancing GOP in CTC-Based Mispronunciation Detection with Phonological Knowledge."

## 🎯 Overview

This implementation provides:
- **PP-AF GOP calculation** with both Restricted (RPS) and Unrestricted (UPS) phoneme substitution modes
- **Wav2Vec2-based CTC model** for acoustic modeling (`facebook/wav2vec2-xisr-53-espeak-cv-ft`)
- **Phoneme confusion mapping** based on linguistic knowledge
- **Evaluation metrics** for mispronunciation detection performance

## 🏗️ Architecture

```
├── main.py                 # Main execution script
├── gop_calculators.py      # Core GOP calculation algorithms
├── phoneme_map.py          # Phoneme confusion patterns
├── data_loader.py          # Data loading utilities
├── evaluation.py           # Performance evaluation metrics
└── requirements.txt        # Python dependencies
```

## 📋 Requirements

### System Requirements
- Python 3.7+
- GPU support (optional, but recommended for faster processing)

### Dependencies
Install all required packages:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `torch` - PyTorch for neural network operations
- `transformers` - Hugging Face models
- `phonemizer` - Text-to-phoneme conversion
- `librosa` - Audio processing
- `scikit-learn` - Evaluation metrics
- `soundfile` - Audio file I/O

### Additional Setup

**Install espeak for phonemization** (required by phonemizer):

- **Ubuntu/Debian**: `sudo apt-get install espeak espeak-data`
- **macOS**: `brew install espeak`
- **Windows**: Download from [eSpeak website](http://espeak.sourceforge.net/download.html)

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd Evaluating-Logit-Based-GOP
pip install -r requirements.txt
```

### 2. Run the Demo
```bash
python main.py
```

This will:
1. Download the pre-trained Wav2Vec2 model (first run only)
2. Generate synthetic audio samples for testing
3. Calculate GOP scores using both RPS and UPS modes
4. Display comparative results

### 3. Expected Output
```
Using device: cuda
Loading model: facebook/wav2vec2-xisr-53-espeak-cv-ft...
Model and processor loaded successfully.
Generated 2 mock audio samples in '.mock_data' directory.

==================================================
Processing: 'this is a test' from .mock_data/mock_audio_0.wav
==================================================

--- Running PP-AF GOP (RPS) ---
Phonemes: ð ɪ s ɪ z ə t ɛ s t
GOP Scores (RPS): [2.14, 1.87, 0.92, ...]
Average GOP (RPS): 1.45

--- Running PP-AF GOP (UPS) ---
Phonemes: ð ɪ s ɪ z ə t ɛ s t
GOP Scores (UPS): [1.89, 1.62, 0.78, ...]
Average GOP (UPS): 1.21
```

## 📊 Understanding the Results

### GOP Score Interpretation
- **Higher scores** = Better pronunciation quality
- **Negative scores** = Potential mispronunciation
- **Typical ranges**: 
  - Good pronunciation: 0.5 to 3.0
  - Mispronunciation: -2.0 to 0.5

### Modes Comparison
- **RPS (Restricted Phoneme Substitutions)**: Uses linguistically-informed confusion patterns
- **UPS (Unrestricted Phoneme Substitutions)**: Tests against all possible phonemes
- RPS is typically more conservative and realistic for L2 learner errors

## 🔧 Using Your Own Data

### Audio Requirements
- **Format**: WAV files
- **Sample rate**: 16 kHz
- **Duration**: 1-10 seconds recommended
- **Quality**: Clear speech, minimal background noise

### Data Integration

1. **Modify data_loader.py**:
```python
def load_real_data(data_dir):
    """Load your audio files and transcripts"""
    samples = []
    for audio_file in glob.glob(f"{data_dir}/*.wav"):
        # Read corresponding transcript
        transcript_file = audio_file.replace('.wav', '.txt')
        with open(transcript_file, 'r') as f:
            transcript = f.read().strip()
        
        samples.append({
            "audio_path": audio_file,
            "canonical_transcript": transcript,
            "label": 1  # 1 for correct, 0 for mispronounced
        })
    return samples
```

2. **Update main.py**:
```python
# Replace this line:
mock_data = load_mock_data(num_samples=2)

# With:
real_data = load_real_data("path/to/your/data")
```

## 🧪 Advanced Usage

### Custom Phoneme Confusion Maps
Modify `phoneme_map.py` to add language-specific or learner-specific confusion patterns:

```python
PHONEME_CONFUSION_MAP = {
    'θ': ['s', 't', 'f'],  # Common /θ/ substitutions
    'r': ['l', 'w'],       # r/l confusion for certain L1 backgrounds
    # Add your patterns here
}
```

### Batch Processing
For processing multiple files efficiently:

```python
def batch_process(audio_files, transcripts):
    results = []
    for audio_path, transcript in zip(audio_files, transcripts):
        phonemes, gop_scores = calculate_pp_af_gop(
            model, processor, audio_path, transcript, mode='RPS'
        )
        results.append({
            'file': audio_path,
            'phonemes': phonemes,
            'gop_scores': gop_scores,
            'avg_gop': np.mean(gop_scores)
        })
    return results
```

## 📈 Evaluation and Metrics

The project includes standard classification metrics:

```python
from evaluation import calculate_metrics

# Calculate performance metrics
metrics = calculate_metrics(
    y_true=[1, 0, 1, 0],  # Ground truth labels
    gop_scores=[1.2, -0.5, 0.8, -1.1],  # Your GOP scores
    threshold=0.0  # Classification threshold
)

print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1-Score: {metrics['f1_score']:.3f}")
```

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```python
   # Add to main.py
   torch.cuda.empty_cache()
   ```

2. **Phonemizer Installation Issues**:
   - Ensure espeak is properly installed
   - Try: `python -c "from phonemizer import phonemize; print('OK')"`

3. **Model Download Fails**:
   - Check internet connection
   - Ensure sufficient disk space (~1.2GB for model)

4. **Audio Loading Errors**:
   - Verify audio files are 16kHz WAV format
   - Use `librosa.load(audio_path, sr=16000)` to check

### Performance Optimization

- **GPU Usage**: Ensure PyTorch CUDA is properly installed
- **Batch Processing**: Process multiple files together
- **Model Caching**: Keep model loaded between calls

## 📚 Research Background

This implementation is based on:

**Paper**: "Enhancing GOP in CTC-Based Mispronunciation Detection with Phonological Knowledge"

**Key Contributions**:
- Alignment-free GOP calculation avoiding forced alignment issues
- Phonological knowledge integration through confusion patterns
- Efficient CTC-based acoustic modeling

**Algorithm**: PP-AF GOP uses perturbation-based scoring:
```
GOP(p_i) = min(L_perturbed) - L_original
```

Where perturbations include phoneme deletion and substitution operations.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For issues and questions:
- Check the troubleshooting section above
- Open an issue on GitHub
- Review the original research paper for algorithmic details

## 🙏 Acknowledgments

- Hugging Face for the pre-trained Wav2Vec2 models
- Original paper authors for the PP-AF GOP methodology
- eSpeak project for phonemization capabilities