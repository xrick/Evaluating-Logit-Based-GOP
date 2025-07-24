import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from phonemizer import phonemize
from .phoneme_map import get_restricted_substitutions
import numpy as np

def init_model_and_processor(model_name: str = "facebook/wav2vec2-xisr-53-espeak-cv-ft"):
    """
    Initializes and returns the Wav2Vec2 model and processor from Hugging Face.
    
    """
    print(f"Loading model: {model_name}...")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    print("Model and processor loaded successfully.")
    return model, processor

def get_ctc_loss(logits, labels, processor, device):
    """
    Helper function to compute CTC loss.
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
    
    # Filter out special tokens for label processing
    special_tokens = {processor.tokenizer.pad_token, processor.tokenizer.unk_token, processor.tokenizer.bos_token, processor.tokenizer.eos_token}
    
    # Ensure labels are valid tokens in the tokenizer vocab
    valid_labels = [l for l in labels if l in processor.tokenizer.vocab]
    if len(valid_labels) != len(labels):
         print(f"Warning: Some phonemes were not in the tokenizer's vocab. Original: {labels}, Filtered: {valid_labels}")
         if not valid_labels: return torch.tensor(float('inf'))

    target_ids = processor.tokenizer.convert_tokens_to_ids(valid_labels)
    
    input_lengths = torch.full(size=(log_probs.shape[1],), fill_value=log_probs.shape[0], dtype=torch.long).to(device)
    target_lengths = torch.full(size=(log_probs.shape[1],), fill_value=len(target_ids), dtype=torch.long).to(device)
    
    ctc_loss = torch.nn.CTCLoss(blank=processor.tokenizer.pad_token_id, zero_infinity=True)
    loss = ctc_loss(log_probs, torch.tensor([target_ids]).to(device), input_lengths, target_lengths)
    
    return loss

def calculate_pp_af_gop(model, processor, audio_path, canonical_transcript, mode='RPS'):
    """
    Calculates Phoneme-Perturbed Alignment-Free GOP (PP-AF GOP).
    This implementation follows the description in Section 2.3.3 and Equation (4) of the paper[cite: 96, 103].
    
    Args:
        model: The Wav2Vec2 CTC model.
        processor: The Wav2Vec2 processor.
        audio_path (str): Path to the audio file.
        canonical_transcript (str): The expected text transcript.
        mode (str): 'RPS' for Restricted Phoneme Substitutions or 'UPS' for Unrestricted.

    Returns:
        A list of GOP scores for each phoneme in the transcript.
    """
    device = model.device
    
    # 1. Load audio and get logits
    audio_input, sample_rate = librosa.load(audio_path, sr=16000)
    input_values = processor(audio_input, return_tensors="pt", sampling_rate=sample_rate).input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits

    # 2. Get original phoneme sequence
    original_phonemes_str = phonemize(canonical_transcript, backend='espeak', language='en-us', with_stress=True)
    original_phonemes = original_phonemes_str.strip().replace(" ", "").replace("ˌ", "").replace("ˈ", "") # Basic cleaning
    original_phonemes = list(original_phonemes)

    # 3. Calculate original CTC loss (L_original)
    l_original = get_ctc_loss(logits, original_phonemes, processor, device).item()
    if np.isinf(l_original):
        print("Error: Original sequence resulted in infinite loss. Cannot compute GOP.")
        return [float('-inf')] * len(original_phonemes)

    phoneme_gops = []
    
    # 4. Iterate through each phoneme to generate perturbations
    for i, p_i in enumerate(original_phonemes):
        perturbed_losses = []

        # Perturbation 1: Deletion
        # Create a new sequence by omitting one phoneme [cite: 100]
        deleted_sequence = original_phonemes[:i] + original_phonemes[i+1:]
        if deleted_sequence:
            loss_del = get_ctc_loss(logits, deleted_sequence, processor, device).item()
            perturbed_losses.append(loss_del)
            
        # Perturbation 2: Substitution
        if mode == 'RPS':
            # Use the predefined confusion map [cite: 99]
            substitutions = get_restricted_substitutions(p_i)
        else: # 'UPS' mode
            # Use the entire vocabulary except the original phoneme
            full_vocab = processor.tokenizer.get_vocab().keys()
            special_tokens = {processor.tokenizer.pad_token, processor.tokenizer.unk_token, processor.tokenizer.bos_token, processor.tokenizer.eos_token, '|'}
            substitutions = [token for token in full_vocab if token not in special_tokens and token != p_i and len(token)==1]
        
        for sub in substitutions:
            substituted_sequence = original_phonemes[:i] + [sub] + original_phonemes[i+1:]
            loss_sub = get_ctc_loss(logits, substituted_sequence, processor, device).item()
            perturbed_losses.append(loss_sub)
        
        # 5. Find min(L_perturbed)
        if not perturbed_losses:
             min_l_perturbed = float('inf')
        else:
             min_l_perturbed = min(perturbed_losses)

        # 6. Calculate GOP score as per Equation (4) 
        gop_score = min_l_perturbed - l_original
        phoneme_gops.append(gop_score)

    return original_phonemes, phoneme_gops