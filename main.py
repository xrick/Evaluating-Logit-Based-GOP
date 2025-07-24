import torch
from gop_calculators import init_model_and_processor, calculate_pp_af_gop
from data_loader import load_mock_data
import numpy as np

def main():
    """
    Main execution function to run the mispronunciation detection experiment.
    """
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 1. Initialization ---
    # Load the pre-trained model and processor 
    model, processor = init_model_and_processor()
    model.to(device)
    
    # --- 2. Load Data ---
    # Load mock data for demonstration
    mock_data = load_mock_data(num_samples=2)
    
    # --- 3. Process each sample ---
    for sample in mock_data:
        audio_path = sample["audio_path"]
        transcript = sample["canonical_transcript"]
        
        print("\n" + "="*50)
        print(f"Processing: '{transcript}' from {audio_path}")
        print("="*50)
        
        # --- 4. Calculate GOP using different methods ---
        
        # Method 1: PP-AF GOP with Restricted Phoneme Substitutions (RPS)
        print("\n--- Running PP-AF GOP (RPS) ---")
        try:
            phonemes_rps, gops_rps = calculate_pp_af_gop(model, processor, audio_path, transcript, mode='RPS')
            print("Phonemes:", " ".join(phonemes_rps))
            print("GOP Scores (RPS):", [round(g, 2) for g in gops_rps])
            print(f"Average GOP (RPS): {np.mean(gops_rps):.2f}")
        except Exception as e:
            print(f"An error occurred during RPS calculation: {e}")

        # Method 2: PP-AF GOP with Unrestricted Phoneme Substitutions (UPS)
        print("\n--- Running PP-AF GOP (UPS) ---")
        try:
            phonemes_ups, gops_ups = calculate_pp_af_gop(model, processor, audio_path, transcript, mode='UPS')
            print("Phonemes:", " ".join(phonemes_ups))
            print("GOP Scores (UPS):", [round(g, 2) for g in gops_ups])
            print(f"Average GOP (UPS): {np.mean(gops_ups):.2f}")
        except Exception as e:
            print(f"An error occurred during UPS calculation: {e}")
            
    print("\n" + "="*50)
    print("Experiment finished.")
    print("="*50)
    print("Note: A higher GOP score indicates a better pronunciation. Negative scores may indicate mispronunciation.")


if __name__ == "__main__":
    main()