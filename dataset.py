import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import soundfile as sf
from torch.utils.data import Dataset

# =====================================================================
# 1. Dynamic Secondary Path Convolution
# =====================================================================

def apply_dynamic_path(signal_batch, path_batch):
    """
    Dynamic Convolution Function (Physical Phase Aligned).
    
    Applies the secondary acoustic path (impulse response) to the predicted 
    anti-noise signal. 
    
    Note: PyTorch's F.conv1d performs cross-correlation natively. To 
    accurately simulate a true causal forward acoustic convolution (as it 
    occurs in physical space), the impulse response filter must be flipped 
    along the time dimension prior to the operation.
    """
    B, T = signal_batch.shape
    L = path_batch.shape[1]
    
    signal_reshaped = signal_batch.view(1, B, T) 
    
    # Flip the secondary path to align with the physical convolution direction
    path_flipped = torch.flip(path_batch, dims=[1])
    path_reshaped = path_flipped.view(B, 1, L)     
    
    pad_len = L - 1
    signal_padded = F.pad(signal_reshaped, (pad_len, 0))
    
    output = F.conv1d(signal_padded, path_reshaped, groups=B) 
    return output.squeeze(0) 

# =====================================================================
# 2. Offline Expected Noise Dataset
# =====================================================================

class PreconvolutedANCDataset(Dataset):
    """
    Dataset class for loading pre-convoluted Active Noise Control data.
    Provides strictly aligned, time-domain triplets for the network:
    [Raw Reference Noise, Secondary Path, Expected Target Noise].
    """
    def __init__(self, dataset_dir, noise_names, path_indices, segment_duration=1.0, sr=48000, is_train=True):
        self.dataset_dir = dataset_dir
        self.noise_names = noise_names
        self.path_indices = path_indices
        self.sr = sr
        self.segment_length = int(segment_duration * sr)
        self.is_train = is_train

        # Load the spatially averaged secondary acoustic paths
        sh_path = os.path.join(dataset_dir, 'sh.npy')
        self.sh_paths = np.load(sh_path).T 
        
        self.expected_dir = os.path.join(dataset_dir, 'EXPECTED_NOISE')
        self.raw_noise_dir = os.path.join(dataset_dir, 'NOISE')

    def __len__(self):
        return len(self.path_indices)

    def _fast_read_slice(self, filepath, start_idx):
        """ 
        High-speed audio segment extraction using Soundfile pointers.
        Ensures memory-efficient reading without loading the entire audio file.
        """
        y, _ = sf.read(filepath, start=start_idx, frames=self.segment_length, dtype='float32', always_2d=False)
        # Enforce mono channel output if the source file is multi-channel
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        return y

    def __getitem__(self, idx):
        path_idx = self.path_indices[idx]
        sh = self.sh_paths[path_idx]
        
        # Guard interval to bypass initial audio transients
        skip_samples = int(20 * self.sr)
        
        if self.is_train:
            # Training Phase: Randomly sample a noise environment
            chosen_noise = random.choice(self.noise_names)
            
            exp_noise_path = os.path.join(self.expected_dir, f"{chosen_noise}_scene_{path_idx+1:02d}.wav")
            raw_noise_path = os.path.join(self.raw_noise_dir, f"{chosen_noise}.wav")
            
            # Fetch total frames via metadata to avoid full disk I/O
            total_frames = sf.info(raw_noise_path).frames
            max_start = total_frames - self.segment_length
            
            start_idx = np.random.randint(skip_samples, max_start) if max_start > skip_samples else max(0, max_start)
                
            seg_exp = self._fast_read_slice(exp_noise_path, start_idx)
            seg_raw = self._fast_read_slice(raw_noise_path, start_idx)
            
        else:
            # Testing Phase: Deterministic scene transition (splicing logic)
            # Simulates an abrupt acoustic environment change at the midpoint of the sample.
            scene1_name = self.noise_names[0]
            scene2_name = self.noise_names[1] if len(self.noise_names) >= 2 else self.noise_names[0]
            
            exp_s1_path = os.path.join(self.expected_dir, f"{scene1_name}_scene_{path_idx+1:02d}.wav")
            raw_s1_path = os.path.join(self.raw_noise_dir, f"{scene1_name}.wav")
            exp_s2_path = os.path.join(self.expected_dir, f"{scene2_name}_scene_{path_idx+1:02d}.wav")
            raw_s2_path = os.path.join(self.raw_noise_dir, f"{scene2_name}.wav")
            
            half_len = self.segment_length // 2
            
            start1 = skip_samples + idx * half_len
            start2 = skip_samples + idx * half_len
            
            max_s1 = sf.info(raw_s1_path).frames - half_len
            max_s2 = sf.info(raw_s2_path).frames - half_len
            
            # Boundary protection with modulo logic
            if start1 > max_s1: start1 = skip_samples + (start1 % max(1, max_s1 - skip_samples))
            if start2 > max_s2: start2 = skip_samples + (start2 % max(1, max_s2 - skip_samples))
            
            seg_exp_s1 = self._fast_read_slice(exp_s1_path, start1)
            seg_raw_s1 = self._fast_read_slice(raw_s1_path, start1)
            seg_exp_s2 = self._fast_read_slice(exp_s2_path, start2)
            seg_raw_s2 = self._fast_read_slice(raw_s2_path, start2)

            seg_exp = np.concatenate([seg_exp_s1, seg_exp_s2])
            seg_raw = np.concatenate([seg_raw_s1, seg_raw_s2])

        # Zero-padding fallback for dimension safety at the end of audio files
        if len(seg_exp) < self.segment_length:
            seg_exp = np.pad(seg_exp, (0, self.segment_length - len(seg_exp)), 'constant')
        if len(seg_raw) < self.segment_length:
            seg_raw = np.pad(seg_raw, (0, self.segment_length - len(seg_raw)), 'constant')
                

        return torch.tensor(seg_raw, dtype=torch.float32), \
               torch.tensor(sh, dtype=torch.float32), \
               torch.tensor(seg_exp, dtype=torch.float32)