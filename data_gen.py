"""
Generate synthetic audio denoising dataset for testing AudioMamba2.

This script creates clean audio signals and adds various types of noise
to create noisy versions for training/testing audio denoising models.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm


class AudioDatasetGenerator:
    """Generate synthetic audio denoising dataset."""
    
    def __init__(
        self,
        output_dir: str,
        sample_rate: int = 16000,
        duration: float = 30.0,
        num_samples: int = 1000,
        noise_types: List[str] = None,
        snr_range: Tuple[float, float] = (-5.0, 20.0),
        seed: int = 42,
    ):
        """
        Initialize dataset generator.
        
        Args:
            output_dir: Directory to save generated dataset
            sample_rate: Audio sample rate (Hz)
            duration: Duration of each audio clip (seconds)
            num_samples: Number of samples to generate
            noise_types: Types of noise to add ['white', 'pink', 'brown', 'babble']
            snr_range: Range of SNR values (dB) for noise addition
            seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = num_samples
        self.noise_types = noise_types or ['white', 'pink', 'brown', 'babble']
        self.snr_range = snr_range
        self.seed = seed
        
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create output directories
        self.clean_dir = self.output_dir / "clean"
        self.noisy_dir = self.output_dir / "noisy"
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        self.noisy_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Dataset will be saved to: {self.output_dir}")
        print(f"   Clean audio: {self.clean_dir}")
        print(f"   Noisy audio: {self.noisy_dir}")
    
    def generate_clean_signal(self, signal_type: str = 'mixed') -> np.ndarray:
        """
        Generate clean audio signal.
        
        Args:
            signal_type: Type of signal to generate
                - 'sine': Pure sine waves
                - 'chirp': Frequency sweep
                - 'speech': Speech-like signals
                - 'music': Music-like signals
                - 'mixed': Mix of above
        
        Returns:
            Clean audio signal of shape (num_samples,)
        """
        num_samples = int(self.duration * self.sample_rate)
        t = np.linspace(0, self.duration, num_samples)
        
        if signal_type == 'sine':
            # Multiple sine waves at different frequencies
            signal = np.zeros(num_samples)
            for freq in [440, 880, 1320, 1760]:  # A4 and harmonics
                signal += np.sin(2 * np.pi * freq * t) / 4
        
        elif signal_type == 'chirp':
            # Frequency sweep (chirp)
            f0, f1 = 100, 8000  # Start and end frequencies
            signal = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * self.duration)))
        
        elif signal_type == 'speech':
            # Speech-like signal (formants + modulation)
            signal = np.zeros(num_samples)
            formants = [700, 1220, 2600]  # Approximate /a/ vowel
            for f in formants:
                signal += np.sin(2 * np.pi * f * t)
            
            # Add pitch modulation
            f0 = 120  # Fundamental frequency
            pitch = np.sin(2 * np.pi * f0 * t)
            signal = signal * (0.5 + 0.5 * pitch)
            
            # Add amplitude envelope
            envelope = np.sin(2 * np.pi * 2 * t) ** 2  # Syllable-like
            signal = signal * envelope
        
        elif signal_type == 'music':
            # Music-like signal (chord progression)
            signal = np.zeros(num_samples)
            chord_duration = self.duration / 4  # 4 chords
            
            chords = [
                [261.63, 329.63, 392.00],  # C major
                [293.66, 369.99, 440.00],  # D minor
                [349.23, 440.00, 523.25],  # F major
                [392.00, 493.88, 587.33],  # G major
            ]
            
            for i, chord in enumerate(chords):
                start_idx = int(i * chord_duration * self.sample_rate)
                end_idx = int((i + 1) * chord_duration * self.sample_rate)
                t_chord = np.linspace(0, chord_duration, end_idx - start_idx)
                
                chord_signal = np.zeros(end_idx - start_idx)
                for freq in chord:
                    chord_signal += np.sin(2 * np.pi * freq * t_chord) / len(chord)
                
                signal[start_idx:end_idx] = chord_signal
        
        elif signal_type == 'mixed':
            # Mix different signal types
            sine_signal = self.generate_clean_signal('sine')
            speech_signal = self.generate_clean_signal('speech')
            music_signal = self.generate_clean_signal('music')
            
            signal = (sine_signal + speech_signal + music_signal) / 3
        
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")
        
        # Normalize to [-1, 1]
        signal = signal / (np.abs(signal).max() + 1e-8)
        
        return signal.astype(np.float32)
    
    def generate_noise(self, noise_type: str, num_samples: int) -> np.ndarray:
        """
        Generate noise signal.
        
        Args:
            noise_type: Type of noise ('white', 'pink', 'brown', 'babble')
            num_samples: Number of samples
        
        Returns:
            Noise signal of shape (num_samples,)
        """
        if noise_type == 'white':
            # White noise (flat spectrum)
            noise = np.random.randn(num_samples)
        
        elif noise_type == 'pink':
            # Pink noise (1/f spectrum)
            white = np.random.randn(num_samples)
            
            # Apply 1/f filter in frequency domain
            fft = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(num_samples)
            freqs[0] = 1e-10  # Avoid division by zero
            
            # 1/f^0.5 characteristic
            fft = fft / np.sqrt(freqs)
            noise = np.fft.irfft(fft, n=num_samples)
        
        elif noise_type == 'brown':
            # Brown noise (1/f^2 spectrum)
            white = np.random.randn(num_samples)
            
            fft = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(num_samples)
            freqs[0] = 1e-10
            
            # 1/f characteristic
            fft = fft / freqs
            noise = np.fft.irfft(fft, n=num_samples)
        
        elif noise_type == 'babble':
            # Babble noise (multiple speakers)
            num_speakers = 5
            babble = np.zeros(num_samples)
            
            for _ in range(num_speakers):
                # Generate speech-like signal for each "speaker"
                t = np.linspace(0, self.duration, num_samples)
                f0 = np.random.uniform(80, 250)  # Pitch variation
                
                # Random formants
                formants = np.random.uniform(300, 3000, size=3)
                speaker = np.zeros(num_samples)
                
                for f in formants:
                    speaker += np.sin(2 * np.pi * f * t)
                
                # Modulate with pitch
                pitch = np.sin(2 * np.pi * f0 * t)
                speaker = speaker * (0.5 + 0.5 * pitch)
                
                babble += speaker / num_speakers
            
            noise = babble
        
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # Normalize
        noise = noise / (np.abs(noise).max() + 1e-8)
        
        return noise.astype(np.float32)
    
    def add_noise(
        self, 
        clean: np.ndarray, 
        noise_type: str, 
        snr_db: float
    ) -> np.ndarray:
        """
        Add noise to clean signal at specified SNR.
        
        Args:
            clean: Clean signal
            noise_type: Type of noise to add
            snr_db: Signal-to-noise ratio in dB
        
        Returns:
            Noisy signal
        """
        # Generate noise
        noise = self.generate_noise(noise_type, len(clean))
        
        # Calculate noise scaling factor for desired SNR
        clean_power = np.mean(clean ** 2)
        noise_power = np.mean(noise ** 2)
        
        # SNR = 10 * log10(P_signal / P_noise)
        # P_noise = P_signal / 10^(SNR/10)
        snr_linear = 10 ** (snr_db / 10)
        noise_scale = np.sqrt(clean_power / (noise_power * snr_linear))
        
        # Add scaled noise
        noisy = clean + noise_scale * noise
        
        # Clip to [-1, 1]
        noisy = np.clip(noisy, -1.0, 1.0)
        
        return noisy.astype(np.float32)
    
    def generate_dataset(self):
        """Generate complete dataset."""
        print(f"\n🎵 Generating {self.num_samples} audio samples...")
        print(f"   Duration: {self.duration}s")
        print(f"   Sample rate: {self.sample_rate}Hz")
        print(f"   Noise types: {self.noise_types}")
        print(f"   SNR range: {self.snr_range[0]} to {self.snr_range[1]} dB\n")
        
        metadata = []
        
        for idx in tqdm(range(self.num_samples), desc="Generating samples"):
            # Choose signal type randomly
            signal_types = ['sine', 'chirp', 'speech', 'music', 'mixed']
            signal_type = np.random.choice(signal_types)
            
            # Generate clean signal
            clean = self.generate_clean_signal(signal_type)
            
            # Choose noise type randomly
            noise_type = np.random.choice(self.noise_types)
            
            # Choose SNR randomly
            snr_db = np.random.uniform(*self.snr_range)
            
            # Add noise
            noisy = self.add_noise(clean, noise_type, snr_db)
            
            # Save audio files
            clean_path = self.clean_dir / f"clean_{idx:06d}.wav"
            noisy_path = self.noisy_dir / f"noisy_{idx:06d}.wav"
            
            sf.write(clean_path, clean, self.sample_rate)
            sf.write(noisy_path, noisy, self.sample_rate)
            
            # Store metadata
            metadata.append({
                'id': idx,
                'clean_path': str(clean_path.relative_to(self.output_dir)),
                'noisy_path': str(noisy_path.relative_to(self.output_dir)),
                'signal_type': signal_type,
                'noise_type': noise_type,
                'snr_db': float(snr_db),
                'duration': self.duration,
                'sample_rate': self.sample_rate,
            })
        
        # Save metadata
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Dataset generation complete!")
        print(f"   Total samples: {self.num_samples}")
        print(f"   Metadata saved to: {metadata_path}")
        
        # Generate dataset splits
        self.create_splits(metadata)
    
    def create_splits(self, metadata: List[dict]):
        """Create train/val/test splits."""
        # Shuffle metadata
        np.random.shuffle(metadata)
        
        # Split ratios
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1
        
        train_size = int(len(metadata) * train_ratio)
        val_size = int(len(metadata) * val_ratio)
        
        train_data = metadata[:train_size]
        val_data = metadata[train_size:train_size + val_size]
        test_data = metadata[train_size + val_size:]
        
        # Save splits
        splits = {
            'train': train_data,
            'validation': val_data,
            'test': test_data,
        }
        
        print(f"\n📊 Dataset Splits:")
        for split_name, split_data in splits.items():
            split_path = self.output_dir / f"{split_name}.json"
            with open(split_path, 'w') as f:
                json.dump(split_data, f, indent=2)
            
            print(f"   {split_name}: {len(split_data)} samples → {split_path}")
    
    def generate_stats(self):
        """Generate dataset statistics."""
        print("\n" + "="*60)
        print("📊 Dataset Statistics")
        print("="*60)
        
        # Load metadata
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Signal type distribution
        signal_types = [item['signal_type'] for item in metadata]
        print("\n🎵 Signal Type Distribution:")
        for sig_type in sorted(set(signal_types)):
            count = signal_types.count(sig_type)
            print(f"   {sig_type:10s}: {count:4d} ({count/len(metadata)*100:5.1f}%)")
        
        # Noise type distribution
        noise_types = [item['noise_type'] for item in metadata]
        print("\n🔊 Noise Type Distribution:")
        for noise_type in sorted(set(noise_types)):
            count = noise_types.count(noise_type)
            print(f"   {noise_type:10s}: {count:4d} ({count/len(metadata)*100:5.1f}%)")
        
        # SNR distribution
        snr_values = [item['snr_db'] for item in metadata]
        print("\n📈 SNR Statistics:")
        print(f"   Mean:  {np.mean(snr_values):6.2f} dB")
        print(f"   Std:   {np.std(snr_values):6.2f} dB")
        print(f"   Min:   {np.min(snr_values):6.2f} dB")
        print(f"   Max:   {np.max(snr_values):6.2f} dB")
        
        # Total size
        total_size_mb = sum(
            os.path.getsize(self.output_dir / item['clean_path']) +
            os.path.getsize(self.output_dir / item['noisy_path'])
            for item in metadata
        ) / (1024 ** 2)
        
        print(f"\n💾 Total Dataset Size: {total_size_mb:.2f} MB")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic audio denoising dataset"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/audio_denoising_synthetic',
        help='Output directory for dataset'
    )
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Audio sample rate (Hz)'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=30.0,
        help='Duration of each audio clip (seconds)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1000,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--noise_types',
        type=str,
        nargs='+',
        default=['white', 'pink', 'brown', 'babble'],
        help='Types of noise to add'
    )
    parser.add_argument(
        '--snr_min',
        type=float,
        default=-5.0,
        help='Minimum SNR (dB)'
    )
    parser.add_argument(
        '--snr_max',
        type=float,
        default=20.0,
        help='Maximum SNR (dB)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = AudioDatasetGenerator(
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        duration=args.duration,
        num_samples=args.num_samples,
        noise_types=args.noise_types,
        snr_range=(args.snr_min, args.snr_max),
        seed=args.seed,
    )
    
    # Generate dataset
    generator.generate_dataset()
    
    # Generate statistics
    generator.generate_stats()
    
    print("✅ All done! Dataset is ready for training.")
    print(f"\n💡 Next steps:")
    print(f"   1. Check your data: ls {args.output_dir}")
    print(f"   2. Test loading: python test_dataset_loading.py")
    print(f"   3. Start training: python flame/train.py --job.config_file train_configs/audiomamba2_debug.toml")


if __name__ == '__main__':
    main()