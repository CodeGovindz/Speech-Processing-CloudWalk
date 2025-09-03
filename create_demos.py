"""
Quick Demo Script for Streamlit Integration
Creates sample audio files if they don't exist
"""

import pandas as pd
import numpy as np
import librosa
import io
import soundfile as sf
from pathlib import Path

def create_demo_audio_files():
    """Create demo audio files from the dataset for testing."""
    print("Creating demo audio files from dataset...")
    
    # Load test data
    try:
        test_df = pd.read_parquet('test-00000-of-00001.parquet')
    except FileNotFoundError:
        print("❌ Test dataset not found. Please ensure the parquet files are present.")
        return False
    
    demo_dir = Path('demo_audio')
    demo_dir.mkdir(exist_ok=True)
    
    created_files = 0
    
    # Create one demo file for each digit
    for digit in range(10):
        output_path = demo_dir / f'digit_{digit}_demo.wav'
        
        # Skip if file already exists
        if output_path.exists():
            continue
            
        # Find first sample of this digit
        digit_samples = test_df[test_df['label'] == digit]
        if len(digit_samples) > 0:
            sample = digit_samples.iloc[0]
            audio_dict = sample['audio']
            
            if 'bytes' in audio_dict:
                try:
                    # Extract audio from bytes
                    audio_buffer = io.BytesIO(audio_dict['bytes'])
                    audio_data, sample_rate = sf.read(audio_buffer)
                    
                    # Save as WAV file
                    sf.write(output_path, audio_data, sample_rate)
                    print(f"✅ Created: {output_path}")
                    created_files += 1
                    
                except Exception as e:
                    print(f"❌ Error creating demo file for digit {digit}: {e}")
    
    if created_files > 0:
        print(f"✅ Successfully created {created_files} demo audio files!")
    else:
        print("ℹ️ All demo files already exist.")
    
    return True

if __name__ == "__main__":
    create_demo_audio_files()
