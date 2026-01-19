"""
Inference Script for Neuro-Symbolic LPR System.

This script provides utilities for running inference on trained models,
including single image, batch, and video processing modes.

Usage:
    python inference.py --model checkpoints/best.pth --input image.jpg
    python inference.py --model checkpoints/best.pth --input video.mp4
    python inference.py --model checkpoints/best.pth --input folder/
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

from config import get_default_config, PLATE_LENGTH
from models import NeuroSymbolicLPR
from models.syntax_mask import validate_plate_output


def load_model(
    checkpoint_path: str,
    device: torch.device,
    config=None
) -> NeuroSymbolicLPR:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint.
        device: Device to load model on.
        config: Model configuration.
        
    Returns:
        Loaded model in evaluation mode.
        
    Security Note:
        torch.load() uses pickle which can execute arbitrary code.
        Only load checkpoints from trusted sources.
    """
    if config is None:
        config = get_default_config()
    
    # Security: Validate checkpoint path is a local file
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found or not a local file: {checkpoint_path}. "
            "For security reasons, only local file paths are accepted."
        )
    
    # Security warning for users
    import warnings
    warnings.warn(
        f"Loading checkpoint from '{checkpoint_path}'. "
        "torch.load() uses pickle which can execute arbitrary code. "
        "Only load checkpoints from trusted sources.",
        UserWarning
    )
    
    # Create model with same args as train.py to ensure architecture compatibility
    model = NeuroSymbolicLPR(
        num_frames=config.model.num_frames,
        lr_size=(config.data.lr_height, config.data.lr_width),
        hr_size=(config.data.hr_height, config.data.hr_width),
        use_shared_attention=config.training.use_shared_attention,
        use_deformable_conv=config.training.use_deformable_conv,
    )
    
    # Load checkpoint with strict=False for backward compatibility
    # This allows loading checkpoints from older model versions that may not
    # have all the new components (e.g., shared attention module)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint['model_state_dict'],
        strict=False
    )
    
    # Log any missing or unexpected keys for debugging
    if missing_keys:
        shared_attn_keys = [k for k in missing_keys if 'shared_attention' in k]
        other_keys = [k for k in missing_keys if 'shared_attention' not in k]
        if shared_attn_keys:
            print(f"[INFO] Checkpoint missing {len(shared_attn_keys)} shared_attention weights (randomly initialized)")
        if other_keys:
            print(f"[WARNING] Checkpoint missing {len(other_keys)} other weights: {other_keys[:5]}...")
    if unexpected_keys:
        print(f"[WARNING] Checkpoint has {len(unexpected_keys)} unexpected keys: {unexpected_keys[:5]}...")
    
    model = model.to(device)
    model.eval()
    
    return model


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    num_frames: int = 5,
    frame_noise_std: float = 0.0
) -> torch.Tensor:
    """
    Preprocess image for model input.
    
    Args:
        image: Input image as numpy array (H, W, C).
        target_size: Target size (H, W).
        num_frames: Number of frames to generate.
        frame_noise_std: Standard deviation of noise to add to extra frames.
            Default is 0.0 (deterministic). Set to small value like 0.01
            for multi-frame augmentation during inference.
        
    Returns:
        Preprocessed tensor of shape (1, num_frames, C, H, W).
    """
    # Resize
    img = Image.fromarray(image)
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    
    # Convert to tensor and normalize to [-1, 1]
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = img_array * 2 - 1
    
    # Convert to CHW format
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    
    # Create multiple frames (duplicate, optionally with slight variations)
    frames = []
    for i in range(num_frames):
        if i == 0 or frame_noise_std <= 0.0:
            # First frame is always clean; others are clean if noise disabled
            frames.append(img_tensor.clone())
        else:
            # Add slight noise for variation (opt-in only)
            noise = torch.randn_like(img_tensor) * frame_noise_std
            frames.append(img_tensor + noise)
    
    # Stack frames and add batch dimension
    frames = torch.stack(frames, dim=0)  # (num_frames, C, H, W)
    frames = frames.unsqueeze(0)  # (1, num_frames, C, H, W)
    
    return frames


def postprocess_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor back to numpy image.
    
    Args:
        tensor: Image tensor of shape (C, H, W) or (1, C, H, W).
        
    Returns:
        Image as numpy array (H, W, C).
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Denormalize from [-1, 1] to [0, 255]
    img = (tensor.cpu().numpy() + 1) / 2 * 255
    img = img.transpose(1, 2, 0)  # CHW to HWC
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    return img


def predict_single(
    model: NeuroSymbolicLPR,
    image: np.ndarray,
    device: torch.device,
    config=None
) -> Tuple[str, float, np.ndarray, bool]:
    """
    Run inference on a single image.
    
    Args:
        model: Trained model.
        image: Input image (H, W, C).
        device: Inference device.
        config: Model configuration.
        
    Returns:
        Tuple of (predicted_text, confidence, sr_image, is_mercosul).
    """
    if config is None:
        config = get_default_config()
    
    # Preprocess
    lr_frames = preprocess_image(
        image,
        (config.data.lr_height, config.data.lr_width),
        config.model.num_frames
    )
    lr_frames = lr_frames.to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(lr_frames, return_intermediates=True)
        
        # Get predictions
        layout_prob = torch.sigmoid(outputs['layout_logits']).item()
        is_mercosul = layout_prob > 0.5
        
        # Decode text
        masked_logits = outputs['masked_logits']
        probs = F.softmax(masked_logits, dim=-1)
        confidence = probs.max(dim=-1)[0].mean().item()
        
        # Decode with syntax mask
        texts = model.syntax_mask.decode_to_text(
            outputs['raw_logits'],
            torch.tensor([is_mercosul], device=device)
        )
        predicted_text = texts[0]
        
        # Get super-resolved image
        sr_image = postprocess_image(outputs['hr_image'])
    
    return predicted_text, confidence, sr_image, is_mercosul


def predict_batch(
    model: NeuroSymbolicLPR,
    images: List[np.ndarray],
    device: torch.device,
    config=None,
    batch_size: int = 16
) -> List[Tuple[str, float, bool]]:
    """
    Run inference on a batch of images.
    
    Args:
        model: Trained model.
        images: List of input images.
        device: Inference device.
        config: Model configuration.
        batch_size: Batch size for inference.
        
    Returns:
        List of (predicted_text, confidence, is_mercosul) tuples.
    """
    if config is None:
        config = get_default_config()
    
    results = []
    
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        
        # Preprocess batch
        batch_tensors = []
        for img in batch_images:
            tensor = preprocess_image(
                img,
                (config.data.lr_height, config.data.lr_width),
                config.model.num_frames
            )
            batch_tensors.append(tensor)
        
        batch = torch.cat(batch_tensors, dim=0).to(device)
        
        # Inference
        with torch.no_grad():
            texts, hr_images, is_mercosul = model.inference(batch)
            
            # Compute confidences
            outputs = model(batch, return_intermediates=False)
            probs = F.softmax(outputs['masked_logits'], dim=-1)
            confidences = probs.max(dim=-1)[0].mean(dim=-1).cpu().tolist()
        
        for j, (text, conf, merc) in enumerate(zip(texts, confidences, is_mercosul)):
            results.append((text, conf, merc.item()))
    
    return results


def process_video(
    model: NeuroSymbolicLPR,
    video_path: str,
    device: torch.device,
    output_path: Optional[str] = None,
    config=None,
    frame_skip: int = 1
) -> List[Tuple[int, str, float]]:
    """
    Process a video file and detect license plates.
    
    Args:
        model: Trained model.
        video_path: Path to input video.
        device: Inference device.
        output_path: Path to save annotated video (optional).
        config: Model configuration.
        frame_skip: Process every N-th frame.
        
    Returns:
        List of (frame_number, predicted_text, confidence) tuples.
    """
    if config is None:
        config = get_default_config()
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output video writer
    writer = None
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    results = []
    frame_buffer = []
    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num % frame_skip == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            text, conf, sr_image, is_merc = predict_single(
                model, frame_rgb, device, config
            )
            
            results.append((frame_num, text, conf))
            
            # Annotate frame
            if writer is not None:
                layout_str = "Mercosul" if is_merc else "Brazilian"
                cv2.putText(
                    frame, f"{text} ({layout_str}) [{conf:.2f}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
                writer.write(frame)
        
        frame_num += 1
        
        # Progress
        if frame_num % 100 == 0:
            print(f"Processed {frame_num}/{total_frames} frames")
    
    cap.release()
    if writer is not None:
        writer.release()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Neuro-Symbolic LPR Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input image, video, or folder')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for folder processing')
    parser.add_argument('--frame-noise-std', type=float, default=0.0,
                        help='Noise std for extra frames (0.0=deterministic, 0.01=slight variation)')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    config = get_default_config()
    model = load_model(args.model, device, config)
    print("Model loaded successfully")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Determine input type
    input_path = Path(args.input)
    
    if input_path.is_file():
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Single image
            image = np.array(Image.open(input_path).convert('RGB'))
            text, conf, sr_image, is_merc = predict_single(model, image, device, config)
            
            layout = "Mercosul" if is_merc else "Brazilian"
            valid = validate_plate_output(text, is_merc)
            
            print(f"Prediction: {text}")
            print(f"Layout: {layout}")
            print(f"Confidence: {conf:.4f}")
            print(f"Valid format: {valid}")
            
            # Save super-resolved image
            sr_path = os.path.join(args.output, f"{input_path.stem}_sr.png")
            Image.fromarray(sr_image).save(sr_path)
            print(f"Super-resolved image saved to: {sr_path}")
            
        elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
            # Video
            output_video = os.path.join(args.output, f"{input_path.stem}_annotated.mp4")
            results = process_video(model, str(input_path), device, output_video, config)
            
            print(f"\nProcessed {len(results)} frames")
            print(f"Annotated video saved to: {output_video}")
            
            # Save results
            results_path = os.path.join(args.output, f"{input_path.stem}_results.txt")
            with open(results_path, 'w') as f:
                for frame_num, text, conf in results:
                    f.write(f"{frame_num}\t{text}\t{conf:.4f}\n")
            print(f"Results saved to: {results_path}")
    
    elif input_path.is_dir():
        # Folder of images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        print(f"Processing {len(image_files)} images...")
        
        images = [np.array(Image.open(f).convert('RGB')) for f in image_files]
        results = predict_batch(model, images, device, config, args.batch_size)
        
        # Print and save results
        results_path = os.path.join(args.output, 'batch_results.txt')
        with open(results_path, 'w') as f:
            for (img_file, (text, conf, is_merc)) in zip(image_files, results):
                layout = "M" if is_merc else "B"
                print(f"{img_file.name}: {text} [{layout}] ({conf:.4f})")
                f.write(f"{img_file.name}\t{text}\t{layout}\t{conf:.4f}\n")
        
        print(f"\nResults saved to: {results_path}")
    
    else:
        print(f"Input not found: {args.input}")


if __name__ == '__main__':
    main()
