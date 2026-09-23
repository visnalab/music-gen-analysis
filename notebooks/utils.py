from functools import partial

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoFeatureExtractor, MusicgenForConditionalGeneration
import librosa
import torchaudio
import pickle
import pandas as pd

def save_pickle(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def load_process_bulk_audio(paths, sr=16000):
    waveforms = {}
    for path in paths:
        audio, sr = librosa.load(path, sr=sr)  # Analyze the first 10 seconds
        waveforms[path] = audio, sr
    return waveforms

def process_bulk_music_gen(waveforms, processor=AutoProcessor.from_pretrained("facebook/musicgen-small")):
    # Preprocess the audio and an optional text query to guide the cross-attention analysis
    inputs = {}
    for path, item in waveforms.items():
        wf, sr = item
        input = processor(
            audio=wf,
            sampling_rate=sr,
            text=[""],  # Optional text to analyze cross-attention alignments
            return_tensors="pt"
        )
        inputs[path] = input
    return inputs

def process_bulk_wave2vec(waveforms, feature_extractor=AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")):
    inputs = {}
    for path, item in waveforms.items():
        wf, sr = item
        input = feature_extractor(
            wf,
            sampling_rate=sr,
            return_tensors="pt"
        )
        inputs[path] = input
    return inputs

def load_music_gen_model():
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-small",
        attn_implementation="eager"
    )
    model.eval()
    return model, processor

def shuffle_pos_embed_hook(module, input, output, chunk_size=1):
    chunks = torch.split(output, split_size_or_sections=chunk_size, dim=0)
    perm = torch.randperm(len(chunks), device=output.device)
    return torch.cat([chunks[i] for i in perm], dim=0)

def extract_attentions(model, inputs_dict, chunk_size=None):
    target_layer = model.decoder.model.decoder.embed_positions
    handle = None
    if chunk_size is not None:
        handle = target_layer.register_forward_hook(
            partial(shuffle_pos_embed_hook, chunk_size=chunk_size)
        )

    attentions = {}
    with torch.no_grad():
        for path, inputs in inputs_dict.items():
            audio_features = inputs["input_values"]
            decoder_input_ids = model.audio_encoder.encode(audio_features).audio_codes
            decoder_outputs = model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=None,
                output_attentions=True,
                return_dict=True
            )
            attentions[path] = decoder_outputs.attentions

    if handle is not None:
        handle.remove()

    return attentions

def compute_mad_by_layer(self_attentions, seq_len):
    # 1. Construct the pairwise distance matrix in steps
    num_layers = len(self_attentions)
    num_heads = self_attentions[0].shape[1]
    steps = np.arange(seq_len)
    distance_matrix = np.abs(steps[:, None] - steps[None, :])

    # 2. Convert steps to relative distance
    relative_distance = distance_matrix / seq_len
    distance_tensor = torch.tensor(relative_distance, dtype=torch.float32, device=self_attentions[0].device)

    # 3. Compute Mean Attention Distance in seconds
    mean_distances = np.zeros((num_layers, num_heads))

    for layer_idx in range(num_layers):
        layer_attn = self_attentions[layer_idx][0]

        # Weighted sum of distances in seconds
        weighted_distances = layer_attn * distance_tensor
        token_mean_distances = torch.sum(weighted_distances, dim=-1)
        head_mean_distances = torch.mean(token_mean_distances, dim=-1)

        mean_distances[layer_idx] = head_mean_distances.cpu().numpy()
    return mean_distances

def compute_mad_dict(attentions_dict):
    mad_dict = {}
    for path, attn in attentions_dict.items():
        seq_length = attn[0].shape[-1]
        mad_dict[path] = compute_mad_by_layer(attn, seq_length)
    return mad_dict

def mad_dict_to_dataframe(mad_dict_layer_avgs):
    records = []

    for path, mads in mad_dict_layer_avgs.items():
        genre = path.split("_")[0].split("/")[-1] # Assuming the genre is encoded in the filename like "classical_1.wav"
        piece_id = path.split("_")[1].replace(".wav", "")  # Extract the piece ID from the filename
        for layer_idx, mad_val in enumerate(mads):  # mads is now a 1D array of layer averages
            records.append({
                "genre": genre,
                "piece_id": piece_id,
                "layer": int(layer_idx),
                "mad": float(mad_val)
            })
    return pd.DataFrame(records)

def compute_relative_attention_entropy(attentions, eps=1e-12):
    """
    attentions shape: [layers, 1, heads, seq_len, seq_len]
    returns: [layers, heads] tensor of mean entropy per head
    """
    seq_length = attentions.shape[-1]
    # Clip values to avoid log(0)
    attns = torch.clamp(attentions, min=eps)
    
    # Calculate row-wise Shannon Entropy: -sum(p * log2(p))
    entropy_per_query = -torch.sum(attns * torch.log2(attns), dim=-1) # Shape: [layers, heads, seq_len]
    
    # Average across all query frames
    mean_entropy = entropy_per_query.mean(dim=-1) / seq_length # Shape: [layers, heads]
    return mean_entropy

def plot_mad_single(mean_distances_seconds, num_heads, num_layers=24):
    plt.figure(figsize=(10, 6))

    # for layer in range(num_layers):
    #     x_coords = [layer] * num_heads
    #     y_coords = mean_distances_seconds[layer]
    #     plt.scatter(x_coords, y_coords, color='forestgreen', alpha=0.4, edgecolors='none')

    layer_averages_sec = np.mean(mean_distances_seconds, axis=1)
    plt.plot(range(num_layers), layer_averages_sec, color='darkorange', linewidth=2.5, marker='o', label='Layer Average')

    plt.title("Mean Attention Distance in Seconds across MusicGen Layers", fontsize=14, fontweight='bold')
    plt.xlabel("Decoder Layer", fontsize=12)
    plt.ylabel("Temporal Attention Distance (Seconds)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    return plt
