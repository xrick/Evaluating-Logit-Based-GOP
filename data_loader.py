import numpy as np
import soundfile as sf
import os

def load_mock_data(num_samples=2, data_dir=".mock_data"):
    """
    Generates synthetic audio files and returns mock data samples.
    This function simulates a real data loader for demonstration purposes.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    mock_samples = [
        {"transcript": "this is a test", "is_correct": True},
        {"transcript": "world hello", "is_correct": False} # Intentionally swapped to simulate error
    ]

    output_data = []
    for i, sample in enumerate(mock_samples[:num_samples]):
        # Generate a simple sine wave as mock audio
        sr = 16000  # Sample rate
        duration = 2  # seconds
        frequency = 440.0 + i * 50 # Vary frequency for different files
        t = np.linspace(0., duration, int(sr * duration), endpoint=False)
        amplitude = np.iinfo(np.int16).max * 0.5
        data = amplitude * np.sin(2. * np.pi * frequency * t)
        
        audio_path = os.path.join(data_dir, f"mock_audio_{i}.wav")
        sf.write(audio_path, data.astype(np.int16), sr)
        
        output_data.append({
            "audio_path": audio_path,
            "canonical_transcript": sample["transcript"],
            "label": 1 if sample["is_correct"] else 0 # 1 for correct, 0 for mispronounced
        })
        
    print(f"Generated {len(output_data)} mock audio samples in '{data_dir}' directory.")
    return output_data