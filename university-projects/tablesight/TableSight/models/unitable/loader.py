#!/usr/bin/env python3
"""
loader.py - Model Loading Helper for UniTable Streamlit Integration

This module provides functions to load UniTable models for inference.
It can be directly copied into your Streamlit application.

Usage in Streamlit:
    from loader import load_unitable_models
    
    # Load all three models
    structure_model, bbox_model, content_model, vocab_s, vocab_b, vocab_c = load_unitable_models(
        base_dir="/path/to/unitable_bundle"
    )

Version: 1.0
Date: 2026-05-19
"""

import torch
import torch.nn as nn
import tokenizers as tk
from pathlib import Path
from functools import partial
from typing import Tuple, Optional


def load_model_from_state(
    state_dict: dict,
    device: torch.device = None
) -> nn.Module:
    """
    Reconstruct and load a UniTable EncoderDecoder model from state dict.
    
    This function automatically infers the model architecture from the 
    saved state dictionary, so you don't need to know the exact layer counts.
    
    Args:
        state_dict: The model state dictionary (from torch.load)
        device: Device to load the model on (default: cuda if available, else cpu)
    
    Returns:
        Loaded EncoderDecoder model in eval mode
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Infer architecture from state dict
    d_model = state_dict['token_embed.embedding.weight'].shape[1]
    vocab_size = state_dict['generator.weight'].shape[0]
    max_seq_len = state_dict['pos_embed.embedding.weight'].shape[0]
    padding_idx = 2
    
    # Count encoder/decoder layers
    enc_keys = [k for k in state_dict if k.startswith('encoder.encoder.layers.')]
    dec_keys = [k for k in state_dict if k.startswith('decoder.decoder.layers.')]
    nlayer_enc = max(int(k.split('.')[3]) for k in enc_keys) + 1 if enc_keys else 2
    nlayer_dec = max(int(k.split('.')[3]) for k in dec_keys) + 1 if dec_keys else 4
    
    # Infer feedforward ratio
    ff_dim = state_dict['encoder.encoder.layers.0.linear1.weight'].shape[0]
    ff_ratio = ff_dim // d_model
    nhead = d_model // 64
    
    # Import UniTable model classes
    # NOTE: You need to have the unitable package in your Python path
    # Add it with: sys.path.insert(0, '/path/to/unitable')
    import sys
    unitable_path = Path(__file__).parent / "source"
    if not unitable_path.exists():
        unitable_path = Path(__file__).parent / "source"
    if unitable_path.exists():
        sys.path.insert(0, str(unitable_path))
    
    from src.model import EncoderDecoder, ImgLinearBackbone, Encoder, Decoder
    
    # Build model
    backbone = ImgLinearBackbone(d_model=d_model, patch_size=16)
    encoder = Encoder(
        d_model=d_model, nhead=nhead, dropout=0.2,
        activation="gelu", norm_first=True,
        nlayer=nlayer_enc, ff_ratio=ff_ratio
    )
    decoder = Decoder(
        d_model=d_model, nhead=nhead, dropout=0.2,
        activation="gelu", norm_first=True,
        nlayer=nlayer_dec, ff_ratio=ff_ratio
    )
    model = EncoderDecoder(
        backbone=backbone, encoder=encoder, decoder=decoder,
        vocab_size=vocab_size, d_model=d_model,
        padding_idx=padding_idx, max_seq_len=max_seq_len,
        dropout=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


def load_tokenizer(vocab_path: Path) -> tk.Tokenizer:
    """
    Load a Tokenizers tokenizer from file.
    
    Args:
        vocab_path: Path to the vocab JSON file
    
    Returns:
        Loaded tokenizer
    """
    return tk.Tokenizer.from_file(str(vocab_path))


def load_unitable_models(
    base_dir: str = ".",
    use_cpu: bool = False
) -> Tuple[nn.Module, nn.Module, nn.Module, tk.Tokenizer, tk.Tokenizer, tk.Tokenizer]:
    """
    Load all three UniTable models and tokenizers for inference.
    
    This is the main convenience function for Streamlit integration.
    Call this once at app startup to load all models.
    
    Args:
        base_dir: Path to the unitable_bundle directory
        use_cpu: If True, load models on CPU (default: use GPU if available)
    
    Returns:
        Tuple of (structure_model, bbox_model, content_model, vocab_s, vocab_b, vocab_c)
        
    Example:
        >>> structure_model, bbox_model, content_model, vocab_s, vocab_b, vocab_c = load_unitable_models(
        ...     base_dir="./unitable_bundle",
        ...     use_cpu=False
        ... )
        >>> print("All models loaded successfully!")
    """
    base_path = Path(base_dir)
    device = torch.device("cpu") if use_cpu else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    print(f"[INFO] Loading UniTable models on {device}...")
    
    # ── Load Vocabularies ──────────────────────────────────────
    vocab_dir = base_path / "vocab"
    print(f"[INFO] Loading tokenizers from {vocab_dir}...")
    
    vocab_c = load_tokenizer(vocab_dir / "vocab_cell_6k.json")
    vocab_b = load_tokenizer(vocab_dir / "vocab_bbox.json")
    vocab_s = load_tokenizer(vocab_dir / "vocab_html.json")
    
    print(f"  vocab_s (HTML): {vocab_s.get_vocab_size()} tokens")
    print(f"  vocab_b (BBox): {vocab_b.get_vocab_size()} tokens")
    print(f"  vocab_c (Cell): {vocab_c.get_vocab_size()} tokens")
    
    # ── Load Models ───────────────────────────────────────────
    models_dir = base_path / "models"
    checkpoints_dir = base_path / "checkpoints"
    
    # Load Structure Model
    print(f"\n[INFO] Loading Structure model...")
    structure_state = torch.load(
        str(models_dir / "unitable_large_structure.pt"),
        map_location=device,
        weights_only=False
    )
    structure_model = load_model_from_state(structure_state, device)
    print("  ✅ Structure model loaded")
    
    # Load BBox Model
    print(f"[INFO] Loading BBox model...")
    bbox_state = torch.load(
        str(models_dir / "unitable_large_bbox.pt"),
        map_location=device,
        weights_only=False
    )
    bbox_model = load_model_from_state(bbox_state, device)
    print("  ✅ BBox model loaded")
    
    # Load Fine-tuned Content Model (Merged)
    print(f"[INFO] Loading Fine-tuned Content model (merged)...")
    content_state = torch.load(
        str(checkpoints_dir / "content_best_merged.pt"),
        map_location=device,
        weights_only=False
    )
    content_model = load_model_from_state(content_state, device)
    print("  ✅ Fine-tuned Content model loaded")
    
    print(f"\n[INFO] All models loaded successfully!")
    print(f"  - Structure model: {models_dir / 'unitable_large_structure.pt'}")
    print(f"  - BBox model: {models_dir / 'unitable_large_bbox.pt'}")
    print(f"  - Content model (FT): {checkpoints_dir / 'content_best_merged.pt'}")
    
    return structure_model, bbox_model, content_model, vocab_s, vocab_b, vocab_c


def load_unitable_models_with_original_content(
    base_dir: str = ".",
    use_cpu: bool = False
) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module, tk.Tokenizer, tk.Tokenizer, tk.Tokenizer]:
    """
    Load all three UniTable models PLUS the original content model.
    
    Use this if you want to compare original vs fine-tuned models.
    
    Returns:
        Tuple of (structure_model, bbox_model, content_model_ft, content_model_orig, vocab_s, vocab_b, vocab_c)
    """
    base_path = Path(base_dir)
    device = torch.device("cpu") if use_cpu else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Load the main models
    structure_model, bbox_model, content_model_ft, vocab_s, vocab_b, vocab_c = \
        load_unitable_models(base_dir, use_cpu)
    
    # Load original content model
    print(f"\n[INFO] Loading original Content model for comparison...")
    orig_state = torch.load(
        str(base_path / "models" / "unitable_large_content.pt"),
        map_location=device,
        weights_only=False
    )
    content_model_orig = load_model_from_state(orig_state, device)
    print("  ✅ Original Content model loaded")
    
    return structure_model, bbox_model, content_model_ft, content_model_orig, vocab_s, vocab_b, vocab_c