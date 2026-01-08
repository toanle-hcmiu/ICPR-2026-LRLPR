#!/usr/bin/env python3
"""
Dataset Splitting Script for ICPR-2026-LRLPR.

Splits the training data into train/val/test sets while preserving
the directory structure (Scenario-A/B, Brazilian/Mercosur).

Usage:
    python scripts/split_dataset.py --data-dir data/train --output-dir data --val-ratio 0.1 --test-ratio 0.1
"""

import os
import argparse
import shutil
import random
from pathlib import Path
from collections import defaultdict


def get_track_info(track_dir: Path) -> dict:
    """Extract info about a track directory."""
    parts = track_dir.relative_to(track_dir.parents[2]).parts
    return {
        'scenario': parts[0],  # Scenario-A or Scenario-B
        'layout': parts[1],    # Brazilian or Mercosur
        'track_name': parts[2],
        'full_path': track_dir
    }


def split_dataset(
    data_dir: str,
    output_dir: str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    copy_files: bool = False
):
    """
    Split dataset into train/val/test sets.
    
    Args:
        data_dir: Source directory containing the training data
        output_dir: Output directory for split datasets
        val_ratio: Ratio of data for validation set
        test_ratio: Ratio of data for test set
        seed: Random seed for reproducibility
        copy_files: If True, copy files; if False, move files (faster, saves space)
    """
    random.seed(seed)
    
    data_path = Path(data_dir).resolve()
    output_path = Path(output_dir).resolve()
    
    # Check if data_dir is already the train folder within output_dir
    # In that case, we need to handle it differently
    if data_path.parent == output_path and data_path.name == 'train':
        print("Source is already 'train' folder inside output directory.")
        print("Creating val and test splits by moving tracks from train...")
        in_place_split = True
    else:
        in_place_split = False
    
    # Find all track directories
    tracks = []
    for scenario_dir in data_path.iterdir():
        if not scenario_dir.is_dir() or not scenario_dir.name.startswith('Scenario'):
            continue
        
        for layout_dir in scenario_dir.iterdir():
            if not layout_dir.is_dir():
                continue
            
            for track_dir in layout_dir.iterdir():
                if track_dir.is_dir() and track_dir.name.startswith('track_'):
                    tracks.append(get_track_info(track_dir))
    
    print(f"Found {len(tracks)} tracks total")
    
    # Group tracks by scenario and layout for stratified splitting
    groups = defaultdict(list)
    for track in tracks:
        key = (track['scenario'], track['layout'])
        groups[key].append(track)
    
    # Print group statistics
    print("\nTracks per group:")
    for key, group_tracks in sorted(groups.items()):
        print(f"  {key[0]}/{key[1]}: {len(group_tracks)}")
    
    # Split each group
    train_tracks = []
    val_tracks = []
    test_tracks = []
    
    for key, group_tracks in groups.items():
        random.shuffle(group_tracks)
        
        n_total = len(group_tracks)
        n_test = max(1, int(n_total * test_ratio))
        n_val = max(1, int(n_total * val_ratio))
        n_train = n_total - n_test - n_val
        
        test_tracks.extend(group_tracks[:n_test])
        val_tracks.extend(group_tracks[n_test:n_test + n_val])
        train_tracks.extend(group_tracks[n_test + n_val:])
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_tracks)}")
    print(f"  Val:   {len(val_tracks)}")
    print(f"  Test:  {len(test_tracks)}")
    
    # Create output directories and move/copy files
    if in_place_split:
        # Only move val and test tracks out of train folder
        splits = {
            'val': val_tracks,
            'test': test_tracks
        }
    else:
        splits = {
            'train': train_tracks,
            'val': val_tracks,
            'test': test_tracks
        }
    
    for split_name, split_tracks in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        for track in split_tracks:
            # Create destination path
            src_path = track['full_path']
            rel_path = src_path.relative_to(data_path)
            dst_path = output_path / split_name / rel_path
            
            # Skip if source and destination are the same
            if src_path.resolve() == dst_path.resolve():
                continue
            
            # Create parent directories
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move or copy the track directory
            if dst_path.exists():
                shutil.rmtree(dst_path)
            
            if copy_files:
                shutil.copytree(src_path, dst_path)
            else:
                shutil.move(str(src_path), str(dst_path))
        
        print(f"  Processed {len(split_tracks)} tracks to {output_path / split_name}")
    
    # Clean up empty directories in source if we moved files
    if not copy_files and not in_place_split:
        for scenario_dir in data_path.iterdir():
            if scenario_dir.is_dir():
                for layout_dir in scenario_dir.iterdir():
                    if layout_dir.is_dir() and not any(layout_dir.iterdir()):
                        layout_dir.rmdir()
                if not any(scenario_dir.iterdir()):
                    scenario_dir.rmdir()
    
    print("\nDone!")
    print(f"\nDataset structure:")
    if in_place_split:
        # Count remaining tracks in train
        remaining_train = sum(1 for _ in data_path.glob('**/track_*'))
        print(f"  {data_path}/  ({remaining_train} tracks remaining)")
    else:
        print(f"  {output_path}/train/  ({len(train_tracks)} tracks)")
    print(f"  {output_path}/val/    ({len(val_tracks)} tracks)")
    print(f"  {output_path}/test/   ({len(test_tracks)} tracks)")


def main():
    parser = argparse.ArgumentParser(description='Split LPR dataset into train/val/test')
    parser.add_argument('--data-dir', type=str, default='data/train',
                        help='Source data directory')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory for splits')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='Test set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--copy', action='store_true',
                        help='Copy files instead of moving (slower, uses more space)')
    
    args = parser.parse_args()
    
    split_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        copy_files=args.copy
    )


if __name__ == '__main__':
    main()
