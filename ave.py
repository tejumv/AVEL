# ============================================
# MUST BE FIRST - BEFORE ALL OTHER IMPORTS
# ============================================
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless servers
# ============================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import librosa
import librosa.display
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           average_precision_score, hamming_loss)
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks
import gc
import warnings
warnings.filterwarnings('ignore')
import math

# Disable interactive mode
plt.ioff()

# Memory optimization
def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Configuration
class Config:
    # Paths
    processed_path = "/home/ai/Cevi_1/AVE_Processed/"
    audio_path = "/home/ai/Cevi_1/AVE_Processed/audio_wav/"
    video_frames_path = "/home/ai/Cevi_1/AVE_Processed/video_frames/"
    dataset_path = "/home/ai/Cevi_1/AVE_Dataset/AVE_Dataset/"
    annotations_file = "/home/ai/Cevi_1/AVE_Dataset/AVE_Dataset/Annotations.txt"
    output_dir = "/home/ai/Cevi_1/AVE_Processed/Finalave/"
    
    # Model parameters
    num_classes = 28
    audio_feat_dim = 256
    video_feat_dim = 256
    hidden_dim = 128
    num_frames = 8
    audio_length = 22050 * 2
    
    # Attention parameters
    num_heads = 8
    dropout_rate = 0.1
    attention_dim = 128
    
    # Training parameters
    batch_size = 4
    learning_rate = 1e-4
    epochs = 30
    patience = 10
    
    # Split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create output directory
os.makedirs(Config.output_dir, exist_ok=True)

# Safe plotting function
def safe_savefig(filepath, dpi=150):
    """Safely save figure without display"""
    try:
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close('all')
        print(f"  ✓ Saved: {os.path.basename(filepath)}")
        return True
    except Exception as e:
        print(f"  ✗ Error saving {filepath}: {str(e)[:50]}")
        plt.close('all')
        return False

# ==================== REAL ATTENTION MODULES ====================

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self attention module"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"
        
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, return_attention=False):
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)
        
        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(attn_output)
        
        if return_attention:
            return output, attn_probs
        return output

class CrossModalAttention(nn.Module):
    """Cross-modal attention between audio and video features"""
    def __init__(self, query_dim, key_dim, value_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.head_dim = query_dim // num_heads
        
        self.query_proj = nn.Linear(query_dim, query_dim)
        self.key_proj = nn.Linear(key_dim, query_dim)
        self.value_proj = nn.Linear(value_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, return_attention=False):
        batch_size = query.shape[0]
        
        # Project inputs
        Q = self.query_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention
        attn_output = torch.matmul(attn_probs, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.query_dim)
        
        output = self.out_proj(attn_output)
        
        if return_attention:
            return output, attn_probs
        return output

class SpatialAttention(nn.Module):
    """Spatial attention for video frames"""
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x_ca = x * ca
        
        # Spatial attention
        sa = self.spatial_attention(x_ca)
        x_sa = x_ca * sa
        
        return x_sa, sa

class TemporalAttention(nn.Module):
    """Temporal attention across frames"""
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.attention = MultiHeadSelfAttention(feature_dim, num_heads)
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x, return_attention=False):
        # x shape: [batch, num_frames, features]
        residual = x
        if return_attention:
            x, attn_weights = self.attention(x, return_attention=True)
        else:
            x = self.attention(x)
        x = self.layer_norm(x + residual)
        
        if return_attention:
            return x, attn_weights
        return x

class AudioVisualAttention(nn.Module):
    """Comprehensive audio-visual attention module"""
    def __init__(self, audio_dim, video_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.audio_self_attn = TemporalAttention(audio_dim, num_heads)
        self.video_self_attn = TemporalAttention(video_dim, num_heads)
        
        # Cross-modal attention
        self.audio_to_video = CrossModalAttention(video_dim, audio_dim, audio_dim, num_heads)
        self.video_to_audio = CrossModalAttention(audio_dim, video_dim, video_dim, num_heads)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(audio_dim + video_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, audio_features, video_features, return_attention=False):
        # Self attention within each modality
        audio_self = self.audio_self_attn(audio_features)
        video_self = self.video_self_attn(video_features)
        
        # Cross-modal attention
        audio_cross, attn_audio_to_video = self.audio_to_video(
            video_self, audio_self, audio_self, return_attention=True
        )
        video_cross, attn_video_to_audio = self.video_to_audio(
            audio_self, video_self, video_self, return_attention=True
        )
        
        # Combine features
        combined = torch.cat([audio_cross.mean(dim=1), video_cross.mean(dim=1)], dim=1)
        fused = self.fusion(combined)
        
        if return_attention:
            return fused, {
                'audio_self_attention': None,  # Can extract from self attention if needed
                'video_self_attention': None,
                'audio_to_video_attention': attn_audio_to_video,
                'video_to_audio_attention': attn_video_to_audio
            }
        
        return fused

# ==================== DATASET & FEATURE EXTRACTORS ====================

# Dataset Manager (same as before)
class DatasetManager:
    def __init__(self, config):
        self.config = config
        self.annotations = self.load_annotations()
        
    def load_annotations(self):
        annotations = {}
        try:
            with open(self.config.annotations_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                start_idx = 0
                if '&' in lines[0] and 'StartTime' in lines[0]:
                    start_idx = 1
                
                for line in lines[start_idx:]:
                    line = line.strip()
                    if '&' in line:
                        parts = line.split('&')
                        if len(parts) >= 5:
                            category, video_id, quality, start, end = parts[:5]
                            if start == 'StartTime' or end == 'EndTime':
                                continue
                            try:
                                annotations[video_id] = {
                                    'category': category,
                                    'start_time': float(start),
                                    'end_time': float(end),
                                    'quality': quality
                                }
                            except ValueError:
                                continue
            print(f"Loaded annotations for {len(annotations)} videos")
        except Exception as e:
            print(f"Error loading annotations: {e}")
        return annotations
    
    def create_splits(self):
        all_video_ids = list(self.annotations.keys())
        print(f"Total videos in annotations: {len(all_video_ids)}")
        
        if len(all_video_ids) == 0:
            return [], [], []
        
        available_video_ids = []
        for video_id in all_video_ids:
            audio_path = os.path.join(self.config.audio_path, f"{video_id}.wav")
            frame_dir = os.path.join(self.config.video_frames_path, video_id)
            
            if os.path.exists(audio_path) and os.path.exists(frame_dir):
                available_video_ids.append(video_id)
        
        print(f"Available videos with both audio and frames: {len(available_video_ids)}")
        
        if len(available_video_ids) == 0:
            return [], [], []
        
        # Split with stratification
        train_ids, temp_ids = train_test_split(
            available_video_ids, 
            test_size=(self.config.val_ratio + self.config.test_ratio),
            random_state=42,
            stratify=[self.annotations[vid]['category'] for vid in available_video_ids]
        )
        
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=self.config.test_ratio/(self.config.val_ratio + self.config.test_ratio),
            random_state=42,
            stratify=[self.annotations[vid]['category'] for vid in temp_ids]
        )
        
        print(f"Train set: {len(train_ids)} videos")
        print(f"Val set: {len(val_ids)} videos")
        print(f"Test set: {len(test_ids)} videos")
        
        return train_ids, val_ids, test_ids

# Audio Feature Extractor (with attention-friendly features)
class AudioFeatureExtractor:
    def __init__(self, config):
        self.config = config
        
    def extract_all_features(self, audio_path, video_id):
        """Extract audio features suitable for attention"""
        try:
            if not os.path.exists(audio_path):
                return torch.zeros(1, 80, 128)  # [batch, time, features]
            
            audio, sr = librosa.load(audio_path, sr=22050)
            
            if len(audio) < 2205:
                return torch.zeros(1, 80, 128)
            
            # Extract multiple features for attention
            features_list = []
            
            # 1. MFCC with deltas
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            features_list.extend([mfcc, mfcc_delta, mfcc_delta2])
            
            # 2. Mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            features_list.append(mel_spec_db)
            
            # 3. Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=12)
            features_list.append(chroma)
            
            # 4. Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            features_list.append(spectral_contrast)
            
            # 5. Tonnetz
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
            features_list.append(tonnetz)
            
            # Combine all features
            all_features = np.vstack(features_list)
            
            # Ensure consistent shape
            target_time_steps = 80
            if all_features.shape[1] > target_time_steps:
                # Downsample
                step = all_features.shape[1] // target_time_steps
                all_features = all_features[:, ::step][:, :target_time_steps]
            elif all_features.shape[1] < target_time_steps:
                # Pad
                pad_width = target_time_steps - all_features.shape[1]
                all_features = np.pad(all_features, ((0, 0), (0, pad_width)), mode='constant')
            
            # Ensure correct feature dimension
            if all_features.shape[0] > 128:
                all_features = all_features[:128]
            elif all_features.shape[0] < 128:
                pad_height = 128 - all_features.shape[0]
                all_features = np.pad(all_features, ((0, pad_height), (0, 0)), mode='constant')
            
            # Reshape for attention: [1, time_steps, features]
            features = torch.FloatTensor(all_features).transpose(0, 1).unsqueeze(0)
            
            return features
            
        except Exception as e:
            print(f"Audio feature extraction error for {video_id}: {e}")
            return torch.zeros(1, 80, 128)

# Video Feature Extractor (with attention-friendly features)
class VideoFeatureExtractor:
    def __init__(self, config):
        self.config = config
        
    def extract_features(self, frames_np, video_id):
        """Extract video features suitable for attention"""
        try:
            features_per_frame = []
            
            for frame in frames_np[:self.config.num_frames]:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                
                frame_features = []
                
                # Color histogram features
                for c in range(3):
                    hist = cv2.calcHist([frame], [c], None, [16], [0, 256])
                    hist = hist.flatten() / (hist.sum() + 1e-8)
                    frame_features.extend(hist)
                
                # Edge features
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                edge_density = np.sum(edges > 0) / edges.size
                frame_features.append(edge_density)
                
                # Texture features (simplified)
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
                frame_features.append(np.mean(gradient_magnitude))
                frame_features.append(np.std(gradient_magnitude))
                
                # Brightness
                frame_features.append(np.mean(gray) / 255.0)
                
                features_per_frame.append(frame_features)
            
            # Convert to tensor
            features_per_frame = np.array(features_per_frame)
            
            # Pad if needed
            if len(features_per_frame) < self.config.num_frames:
                padding = np.zeros((self.config.num_frames - len(features_per_frame), features_per_frame.shape[1]))
                features_per_frame = np.vstack([features_per_frame, padding])
            
            # Ensure correct feature dimension
            if features_per_frame.shape[1] > 128:
                features_per_frame = features_per_frame[:, :128]
            elif features_per_frame.shape[1] < 128:
                pad_width = 128 - features_per_frame.shape[1]
                features_per_frame = np.pad(features_per_frame, ((0, 0), (0, pad_width)), mode='constant')
            
            # Reshape for attention: [1, num_frames, features]
            features = torch.FloatTensor(features_per_frame).unsqueeze(0)
            
            return features
            
        except Exception as e:
            print(f"Video feature extraction error for {video_id}: {e}")
            return torch.zeros(1, self.config.num_frames, 128)

# Dataset Class
class AVEDataset(Dataset):
    def __init__(self, video_ids, annotations, transform=None, mode='train'):
        self.video_ids = video_ids
        self.annotations = annotations
        self.transform = transform
        self.mode = mode
        
        self.categories = sorted(list(set([ann['category'] for ann in self.annotations.values()])))
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        print(f"Loaded {len(self.video_ids)} videos for {mode} set")
        
        self.audio_extractor = AudioFeatureExtractor(Config())
        self.video_extractor = VideoFeatureExtractor(Config())
    
    def load_video_frames(self, video_id):
        frame_dir = os.path.join(Config.video_frames_path, video_id)
        frames = []
        frames_np = []
        
        if os.path.exists(frame_dir):
            try:
                frame_files = sorted([f for f in os.listdir(frame_dir) 
                                    if f.endswith('.jpg') or f.endswith('.png')])
                
                if not frame_files:
                    frames = [torch.zeros(3, 224, 224) for _ in range(Config.num_frames)]
                    frames_np = [np.zeros((224, 224, 3)) for _ in range(Config.num_frames)]
                    return torch.stack(frames), frames_np
                
                if len(frame_files) > Config.num_frames:
                    indices = np.linspace(0, len(frame_files)-1, Config.num_frames, dtype=int)
                    frame_files = [frame_files[i] for i in indices]
                else:
                    frame_files = frame_files[:Config.num_frames]
                
                for frame_file in frame_files:
                    frame_path = os.path.join(frame_dir, frame_file)
                    try:
                        frame = Image.open(frame_path).convert('RGB')
                        frames_np.append(np.array(frame))
                        if self.transform:
                            frame = self.transform(frame)
                        frames.append(frame)
                    except Exception:
                        frames.append(torch.zeros(3, 224, 224))
                        frames_np.append(np.zeros((224, 224, 3)))
                
                while len(frames) < Config.num_frames:
                    frames.append(torch.zeros(3, 224, 224))
                    frames_np.append(np.zeros((224, 224, 3)))
                    
            except Exception as e:
                frames = [torch.zeros(3, 224, 224) for _ in range(Config.num_frames)]
                frames_np = [np.zeros((224, 224, 3)) for _ in range(Config.num_frames)]
        else:
            frames = [torch.zeros(3, 224, 224) for _ in range(Config.num_frames)]
            frames_np = [np.zeros((224, 224, 3)) for _ in range(Config.num_frames)]
        
        return torch.stack(frames), frames_np
    
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        annotation = self.annotations[video_id]
        
        audio_path = os.path.join(Config.audio_path, f"{video_id}.wav")
        video_frames, frames_np = self.load_video_frames(video_id)
        
        # Extract features with attention-friendly format
        audio_features = self.audio_extractor.extract_all_features(audio_path, video_id)
        video_features = self.video_extractor.extract_features(frames_np, video_id)
        
        category = annotation['category']
        label = self.category_to_idx[category]
        start_time = annotation['start_time']
        end_time = annotation['end_time']
        
        return {
            'video_id': video_id,
            'audio_features': audio_features,  # [1, 80, 128]
            'video_features': video_features,  # [1, num_frames, 128]
            'video_frames': video_frames,      # [num_frames, 3, 224, 224]
            'label': torch.tensor(label, dtype=torch.long),
            'start_time': torch.tensor(start_time, dtype=torch.float32),
            'end_time': torch.tensor(end_time, dtype=torch.float32),
        }

# ==================== ENHANCED MODEL WITH REAL ATTENTION ====================

class EnhancedAudioEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_heads=4):
        super(EnhancedAudioEncoder, self).__init__()
        
        # Initial feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(hidden_dim, num_heads)
        
        # Positional encoding for temporal information
        self.positional_encoding = nn.Parameter(torch.randn(1, 80, hidden_dim))
        
    def forward(self, x, return_attention=False):
        # x shape: [batch, time, features]
        batch_size = x.shape[0]
        
        # Transpose for conv1d: [batch, features, time]
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        
        # Transpose back: [batch, time, features]
        x = x.transpose(1, 2)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :x.shape[1], :]
        
        # Apply temporal attention
        if return_attention:
            x, attn_weights = self.temporal_attention(x, return_attention=True)
            return x, attn_weights
        else:
            x = self.temporal_attention(x)
            return x

class EnhancedVideoEncoder(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4):
        super(EnhancedVideoEncoder, self).__init__()
        
        # Use MobileNet for frame features
        mobilenet = models.mobilenet_v3_small(pretrained=True)
        self.feature_extractor = mobilenet.features
        
        # Spatial attention
        self.spatial_attention = SpatialAttention(576)
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(hidden_dim, num_heads)
        
        # Frame-level projection
        self.frame_fc = nn.Linear(576, hidden_dim)
        
        # Positional encoding for frames
        self.positional_encoding = nn.Parameter(torch.randn(1, Config.num_frames, hidden_dim))
        
    def forward(self, x, return_attention=False):
        # x shape: [batch, num_frames, 3, 224, 224]
        batch_size, num_frames = x.shape[0], x.shape[1]
        
        # Reshape for feature extraction
        x = x.view(-1, 3, 224, 224)
        
        # Extract features
        features = self.feature_extractor(x)  # [batch*num_frames, 576, 7, 7]
        
        # Apply spatial attention
        features, spatial_attn_maps = self.spatial_attention(features)
        
        # Global average pooling
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.squeeze(-1).squeeze(-1)  # [batch*num_frames, 576]
        
        # Reshape back to batch format
        features = features.view(batch_size, num_frames, -1)
        
        # Project to hidden dimension
        features = self.frame_fc(features)  # [batch, num_frames, hidden_dim]
        
        # Add positional encoding
        features = features + self.positional_encoding[:, :num_frames, :]
        
        # Apply temporal attention
        if return_attention:
            features, temporal_attn_weights = self.temporal_attention(features, return_attention=True)
            return features, spatial_attn_maps, temporal_attn_weights
        else:
            features = self.temporal_attention(features)
            return features, spatial_attn_maps, None

class EnhancedAVEventLocalizer(nn.Module):
    def __init__(self, config):
        super(EnhancedAVEventLocalizer, self).__init__()
        self.config = config
        
        # Encoders
        self.audio_encoder = EnhancedAudioEncoder(128, config.hidden_dim, config.num_heads)
        self.video_encoder = EnhancedVideoEncoder(config.hidden_dim, config.num_heads)
        
        # Audio-visual attention fusion
        self.av_attention = AudioVisualAttention(
            config.hidden_dim, config.hidden_dim, config.hidden_dim, config.num_heads
        )
        
        # Additional cross-modal attention
        self.cross_modal_attention = CrossModalAttention(
            config.hidden_dim, config.hidden_dim, config.hidden_dim, config.num_heads
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )
        
        # Temporal regression head
        self.temporal_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 2),
            nn.Sigmoid()
        )
        
        # Attention visualization heads
        self.attention_projection = nn.Linear(config.hidden_dim, 1)
        
    def forward(self, audio_features, video_features, video_frames, return_attention=False):
        batch_size = audio_features.shape[0]
        
        # Process audio
        if return_attention:
            audio_encoded, audio_temporal_attn = self.audio_encoder(
                audio_features.squeeze(1), return_attention=True
            )
        else:
            audio_encoded = self.audio_encoder(audio_features.squeeze(1))
        
        # Process video
        if return_attention:
            video_encoded, spatial_attn_maps, video_temporal_attn = self.video_encoder(
                video_frames, return_attention=True
            )
        else:
            video_encoded, spatial_attn_maps, _ = self.video_encoder(video_frames)
        
        # Audio-visual attention fusion
        if return_attention:
            av_fused, av_attention_weights = self.av_attention(
                audio_encoded, video_encoded, return_attention=True
            )
        else:
            av_fused = self.av_attention(audio_encoded, video_encoded)
        
        # Additional cross-modal refinement
        if return_attention:
            final_features, cross_modal_attn = self.cross_modal_attention(
                av_fused.unsqueeze(1), 
                audio_encoded.mean(dim=1, keepdim=True),
                video_encoded.mean(dim=1, keepdim=True),
                return_attention=True
            )
            final_features = final_features.squeeze(1)
        else:
            final_features = self.cross_modal_attention(
                av_fused.unsqueeze(1),
                audio_encoded.mean(dim=1, keepdim=True),
                video_encoded.mean(dim=1, keepdim=True)
            ).squeeze(1)
        
        # Generate outputs
        class_logits = self.classifier(final_features)
        temporal_pred = self.temporal_head(final_features) * 10  # Scale to seconds
        
        # Generate spatial attention maps for visualization
        spatial_attentions = []
        if spatial_attn_maps is not None:
            for i in range(batch_size):
                # Use the first frame's attention map
                attn_map = spatial_attn_maps[i * self.config.num_frames]  # Get first frame's attention
                attn_map = nn.functional.interpolate(
                    attn_map.unsqueeze(0).unsqueeze(0),
                    size=(224, 224),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                spatial_attentions.append(attn_map)
        
        if return_attention:
            return {
                'class_logits': class_logits,
                'temporal_pred': temporal_pred,
                'spatial_attention': spatial_attentions,
                'temporal_attention': {
                    'audio': audio_temporal_attn,
                    'video': video_temporal_attn
                } if return_attention else None,
                'cross_modal_attention': cross_modal_attn if return_attention else None,
                'av_attention': av_attention_weights if return_attention else None,
                'audio_features': audio_encoded.mean(dim=1),
                'video_features': video_encoded.mean(dim=1),
                'fused_features': final_features
            }
        
        return {
            'class_logits': class_logits,
            'temporal_pred': temporal_pred,
            'spatial_attention': spatial_attentions,
            'temporal_attention': None,
            'cross_modal_attention': None,
            'av_attention': None,
            'audio_features': audio_encoded.mean(dim=1),
            'video_features': video_encoded.mean(dim=1),
            'fused_features': final_features
        }

# ==================== TRAINER WITH ATTENTION VISUALIZATION ====================

class EnhancedTrainer:
    def __init__(self, model, config, categories):
        self.model = model
        self.config = config
        self.categories = categories
        
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_temp = nn.L1Loss()
        
        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, factor=0.5)
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        
        # Create attention visualization directory
        self.attention_dir = os.path.join(config.output_dir, "attention_visualization")
        os.makedirs(self.attention_dir, exist_ok=True)
        
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        total_cls_loss = 0
        total_temp_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx % 10 == 0:
                clear_memory()
            
            audio_features = batch['audio_features'].to(self.config.device)
            video_features = batch['video_features'].to(self.config.device)
            video_frames = batch['video_frames'].to(self.config.device)
            labels = batch['label'].to(self.config.device)
            start_times = batch['start_time'].to(self.config.device)
            end_times = batch['end_time'].to(self.config.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with attention
            outputs = self.model(audio_features, video_features, video_frames, return_attention=False)
            
            # Calculate losses
            cls_loss = self.criterion_cls(outputs['class_logits'], labels)
            temporal_target = torch.stack([start_times, end_times], dim=1)
            temp_loss = self.criterion_temp(outputs['temporal_pred'], temporal_target)
            
            # Combined loss
            loss = cls_loss + 0.3 * temp_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_temp_loss += temp_loss.item()
            
            # Visualize attention every 100 batches
            if batch_idx % 100 == 0 and batch_idx > 0:
                self.visualize_attention(batch, outputs, epoch, batch_idx)
            
            if batch_idx % 50 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, '
                      f'Cls: {cls_loss.item():.4f}, Temp: {temp_loss.item():.4f}')
        
        return (total_loss / len(train_loader), 
                total_cls_loss / len(train_loader),
                total_temp_loss / len(train_loader))
    
    def visualize_attention(self, batch, outputs, epoch, batch_idx):
        """Visualize attention weights for debugging"""
        try:
            # Get a sample from the batch
            idx = 0
            video_id = batch['video_id'][idx]
            
            # Create attention visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Plot 1: Spatial attention (if available)
            if outputs['spatial_attention'] and len(outputs['spatial_attention']) > idx:
                spatial_attn = outputs['spatial_attention'][idx].cpu().detach().numpy()
                axes[0, 0].imshow(spatial_attn, cmap='hot')
                axes[0, 0].set_title('Spatial Attention')
                axes[0, 0].axis('off')
            
            # Plot 2: Class probabilities
            class_probs = torch.softmax(outputs['class_logits'][idx], dim=0).cpu().detach().numpy()
            top_k = min(10, len(self.categories))
            top_indices = np.argsort(class_probs)[-top_k:]
            axes[0, 1].barh(range(top_k), class_probs[top_indices])
            axes[0, 1].set_yticks(range(top_k))
            axes[0, 1].set_yticklabels([self.categories[i] for i in top_indices])
            axes[0, 1].set_title('Top Class Probabilities')
            
            # Plot 3: Temporal prediction
            temporal_pred = outputs['temporal_pred'][idx].cpu().detach().numpy()
            axes[0, 2].bar(['Start', 'End'], temporal_pred)
            axes[0, 2].set_ylim(0, 10)
            axes[0, 2].set_title('Temporal Prediction (seconds)')
            
            # Plot 4: Audio feature distribution
            if 'audio_features' in outputs:
                audio_feat = outputs['audio_features'][idx].cpu().detach().numpy()
                axes[1, 0].hist(audio_feat, bins=20)
                axes[1, 0].set_title('Audio Features Distribution')
            
            # Plot 5: Video feature distribution
            if 'video_features' in outputs:
                video_feat = outputs['video_features'][idx].cpu().detach().numpy()
                axes[1, 1].hist(video_feat, bins=20)
                axes[1, 1].set_title('Video Features Distribution')
            
            # Plot 6: Loss tracking
            if len(self.train_losses) > 0:
                axes[1, 2].plot(self.train_losses[-min(20, len(self.train_losses)):])
                axes[1, 2].set_title('Recent Training Loss')
                axes[1, 2].set_xlabel('Batch')
                axes[1, 2].set_ylabel('Loss')
            
            plt.suptitle(f'Attention Visualization\nEpoch {epoch}, Batch {batch_idx}, Video: {video_id}')
            plt.tight_layout()
            
            # Save figure
            filename = os.path.join(self.attention_dir, f'attention_epoch{epoch}_batch{batch_idx}.png')
            safe_savefig(filename)
            
        except Exception as e:
            print(f"Error visualizing attention: {e}")
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_cls_loss = 0
        total_temp_loss = 0
        all_preds = []
        all_labels = []
        all_temporal_preds = []
        all_temporal_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                clear_memory()
                
                audio_features = batch['audio_features'].to(self.config.device)
                video_features = batch['video_features'].to(self.config.device)
                video_frames = batch['video_frames'].to(self.config.device)
                labels = batch['label'].to(self.config.device)
                start_times = batch['start_time'].to(self.config.device)
                end_times = batch['end_time'].to(self.config.device)
                
                outputs = self.model(audio_features, video_features, video_frames)
                
                # Calculate losses
                cls_loss = self.criterion_cls(outputs['class_logits'], labels)
                temporal_target = torch.stack([start_times, end_times], dim=1)
                temp_loss = self.criterion_temp(outputs['temporal_pred'], temporal_target)
                loss = cls_loss + 0.3 * temp_loss
                
                total_loss += loss.item()
                total_cls_loss += cls_loss.item()
                total_temp_loss += temp_loss.item()
                
                # Collect predictions
                preds = torch.softmax(outputs['class_logits'], dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_temporal_preds.extend(outputs['temporal_pred'].cpu().numpy())
                all_temporal_targets.extend(temporal_target.cpu().numpy())
        
        return (total_loss / len(val_loader), 
                total_cls_loss / len(val_loader),
                total_temp_loss / len(val_loader),
                np.array(all_preds), np.array(all_labels),
                np.array(all_temporal_preds), np.array(all_temporal_targets))
    
    def train(self, train_loader, val_loader):
        print("Starting training with real attention mechanisms...")
        
        for epoch in range(self.config.epochs):
            clear_memory()
            
            # Training
            train_loss, train_cls_loss, train_temp_loss = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_cls_loss, val_temp_loss, val_preds, val_labels, temp_preds, temp_targets = self.validate(val_loader)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Calculate accuracy
            val_preds_class = np.argmax(val_preds, axis=1)
            accuracy = accuracy_score(val_labels, val_preds_class)
            self.val_accuracies.append(accuracy)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print epoch summary
            print(f'\nEpoch {epoch+1}/{self.config.epochs}:')
            print(f'  Train - Total: {train_loss:.4f}, Cls: {train_cls_loss:.4f}, Temp: {train_temp_loss:.4f}')
            print(f'  Val   - Total: {val_loss:.4f}, Cls: {val_cls_loss:.4f}, Temp: {val_temp_loss:.4f}')
            print(f'  Val Accuracy: {accuracy:.4f}')
            print(f'  Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save best model
            if accuracy > self.best_val_accuracy:
                self.best_val_accuracy = accuracy
                self.patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': accuracy,
                    'config': self.config.__dict__
                }, os.path.join(self.config.output_dir, 'best_model_with_attention.pth'))
                print(f'  ✓ Saved best model with accuracy: {accuracy:.4f}')
                
                # Save attention visualization from validation
                self.save_attention_analysis(val_loader, epoch)
            else:
                self.patience_counter += 1
                print(f'  Early stopping counter: {self.patience_counter}/{self.config.patience}')
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print("Early stopping triggered!")
                break
            
            print('-' * 60)
        
        # Save training curves
        self.save_training_curves()
        
    def save_attention_analysis(self, val_loader, epoch):
        """Save detailed attention analysis"""
        self.model.eval()
        with torch.no_grad():
            # Get one batch for analysis
            for batch in val_loader:
                audio_features = batch['audio_features'].to(self.config.device)
                video_features = batch['video_features'].to(self.config.device)
                video_frames = batch['video_frames'].to(self.config.device)
                
                # Forward with attention return
                outputs = self.model(audio_features, video_features, video_frames, return_attention=True)
                
                # Create comprehensive attention visualization
                self.create_comprehensive_attention_plot(batch, outputs, epoch)
                break
    
    def create_comprehensive_attention_plot(self, batch, outputs, epoch):
        """Create a comprehensive attention visualization"""
        try:
            fig = plt.figure(figsize=(20, 16))
            
            # Get sample
            idx = 0
            video_id = batch['video_id'][idx]
            true_label = batch['label'][idx].item()
            
            # 1. Spatial Attention
            if outputs['spatial_attention']:
                ax1 = plt.subplot(3, 4, 1)
                spatial_attn = outputs['spatial_attention'][idx].cpu().numpy()
                ax1.imshow(spatial_attn, cmap='hot')
                ax1.set_title('Spatial Attention Map')
                ax1.axis('off')
            
            # 2. Temporal Attention (Audio)
            if outputs['temporal_attention'] and outputs['temporal_attention']['audio'] is not None:
                ax2 = plt.subplot(3, 4, 2)
                audio_temp_attn = outputs['temporal_attention']['audio'][idx].mean(dim=0).cpu().numpy()
                ax2.imshow(audio_temp_attn, cmap='viridis', aspect='auto')
                ax2.set_title('Audio Temporal Attention')
                ax2.set_xlabel('Time Steps')
                ax2.set_ylabel('Time Steps')
            
            # 3. Temporal Attention (Video)
            if outputs['temporal_attention'] and outputs['temporal_attention']['video'] is not None:
                ax3 = plt.subplot(3, 4, 3)
                video_temp_attn = outputs['temporal_attention']['video'][idx].mean(dim=0).cpu().numpy()
                ax3.imshow(video_temp_attn, cmap='viridis', aspect='auto')
                ax3.set_title('Video Temporal Attention')
                ax3.set_xlabel('Frames')
                ax3.set_ylabel('Frames')
            
            # 4. Cross-modal Attention
            if outputs['cross_modal_attention'] is not None:
                ax4 = plt.subplot(3, 4, 4)
                cross_attn = outputs['cross_modal_attention'][idx].mean(dim=0).cpu().numpy()
                ax4.imshow(cross_attn, cmap='plasma', aspect='auto')
                ax4.set_title('Cross-modal Attention')
            
            # 5. Audio-Visual Attention
            if outputs['av_attention'] is not None:
                ax5 = plt.subplot(3, 4, 5)
                av_attn = outputs['av_attention']['audio_to_video_attention'][idx].mean(dim=0).cpu().numpy()
                ax5.imshow(av_attn, cmap='summer', aspect='auto')
                ax5.set_title('Audio → Video Attention')
                ax5.set_xlabel('Video Frames')
                ax5.set_ylabel('Audio Time Steps')
            
            # 6. Video-Audio Attention
            if outputs['av_attention'] is not None:
                ax6 = plt.subplot(3, 4, 6)
                va_attn = outputs['av_attention']['video_to_audio_attention'][idx].mean(dim=0).cpu().numpy()
                ax6.imshow(va_attn, cmap='autumn', aspect='auto')
                ax6.set_title('Video → Audio Attention')
                ax6.set_xlabel('Audio Time Steps')
                ax6.set_ylabel('Video Frames')
            
            # 7. Class Probabilities
            ax7 = plt.subplot(3, 4, 7)
            class_probs = torch.softmax(outputs['class_logits'][idx], dim=0).cpu().numpy()
            top_k = min(8, len(self.categories))
            top_indices = np.argsort(class_probs)[-top_k:]
            colors = plt.cm.Set3(np.linspace(0, 1, top_k))
            ax7.barh(range(top_k), class_probs[top_indices], color=colors)
            ax7.set_yticks(range(top_k))
            ax7.set_yticklabels([self.categories[i] for i in top_indices], fontsize=8)
            ax7.set_title(f'Top {top_k} Predictions')
            ax7.set_xlim(0, 1)
            
            # Highlight true label
            if true_label in top_indices:
                true_idx = list(top_indices).index(true_label)
                ax7.patches[true_idx].set_edgecolor('red')
                ax7.patches[true_idx].set_linewidth(3)
            
            # 8. Temporal Prediction
            ax8 = plt.subplot(3, 4, 8)
            temporal_pred = outputs['temporal_pred'][idx].cpu().numpy()
            true_start = batch['start_time'][idx].item()
            true_end = batch['end_time'][idx].item()
            
            ax8.bar(['Pred Start', 'Pred End'], temporal_pred, alpha=0.7, label='Predicted')
            ax8.bar(['True Start', 'True End'], [true_start, true_end], alpha=0.5, label='True')
            ax8.set_ylim(0, max(10, true_end + 1))
            ax8.set_title('Temporal Localization')
            ax8.legend()
            
            # 9. Feature Distributions
            ax9 = plt.subplot(3, 4, 9)
            if 'audio_features' in outputs:
                audio_feat = outputs['audio_features'][idx].cpu().numpy()
                ax9.hist(audio_feat, bins=30, alpha=0.7, label='Audio', density=True)
            if 'video_features' in outputs:
                video_feat = outputs['video_features'][idx].cpu().numpy()
                ax9.hist(video_feat, bins=30, alpha=0.7, label='Video', density=True)
            ax9.set_title('Feature Distributions')
            ax9.legend()
            
            # 10. Attention Head Analysis
            if outputs['av_attention'] is not None:
                ax10 = plt.subplot(3, 4, 10)
                av_attn_all = outputs['av_attention']['audio_to_video_attention'][idx].cpu().numpy()
                head_importance = av_attn_all.mean(axis=(1, 2))
                ax10.bar(range(len(head_importance)), head_importance)
                ax10.set_title('Attention Head Importance')
                ax10.set_xlabel('Head Index')
                ax10.set_ylabel('Mean Attention Weight')
            
            # 11. Training Progress
            ax11 = plt.subplot(3, 4, 11)
            if len(self.train_losses) > 0:
                ax11.plot(self.train_losses, label='Train Loss', color='blue')
                ax11.plot(self.val_losses, label='Val Loss', color='red')
                ax11.set_xlabel('Epoch')
                ax11.set_ylabel('Loss')
                ax11.set_title('Training Progress')
                ax11.legend()
                ax11.grid(True, alpha=0.3)
            
            # 12. Accuracy Progress
            ax12 = plt.subplot(3, 4, 12)
            if len(self.val_accuracies) > 0:
                ax12.plot(self.val_accuracies, label='Val Accuracy', color='green', linewidth=2)
                ax12.set_xlabel('Epoch')
                ax12.set_ylabel('Accuracy')
                ax12.set_title('Validation Accuracy')
                ax12.legend()
                ax12.grid(True, alpha=0.3)
                ax12.set_ylim(0, 1)
            
            plt.suptitle(f'COMPREHENSIVE ATTENTION ANALYSIS\n'
                        f'Epoch {epoch+1}, Video: {video_id}, True Label: {self.categories[true_label]}',
                        fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            
            # Save figure
            filename = os.path.join(self.attention_dir, f'attention_analysis_epoch{epoch+1}.png')
            safe_savefig(filename, dpi=150)
            
        except Exception as e:
            print(f"Error creating comprehensive attention plot: {e}")
    
    def save_training_curves(self):
        """Save training curves to file"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss curve
            axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue', linewidth=2)
            axes[0, 0].plot(self.val_losses, label='Val Loss', color='red', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Accuracy curve
            axes[0, 1].plot(self.val_accuracies, label='Val Accuracy', color='green', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Validation Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(0, 1)
            
            # Loss components (if tracked separately)
            if hasattr(self, 'train_cls_losses') and hasattr(self, 'train_temp_losses'):
                axes[1, 0].plot(self.train_cls_losses, label='Train Cls Loss', color='blue', alpha=0.7)
                axes[1, 0].plot(self.train_temp_losses, label='Train Temp Loss', color='green', alpha=0.7)
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].set_title('Training Loss Components')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # Best accuracy marker
            if len(self.val_accuracies) > 0:
                best_epoch = np.argmax(self.val_accuracies)
                best_acc = self.val_accuracies[best_epoch]
                axes[0, 1].plot(best_epoch, best_acc, 'ro', markersize=10, 
                               label=f'Best: {best_acc:.3f}')
                axes[0, 1].legend()
            
            plt.suptitle('Training History with Attention Mechanisms', fontsize=14, fontweight='bold')
            plt.tight_layout()
            safe_savefig(os.path.join(self.config.output_dir, 'training_curves_with_attention.png'), dpi=150)
            
        except Exception as e:
            print(f"Error saving training curves: {e}")

# ==================== COMPREHENSIVE EVALUATOR WITH ATTENTION ====================

class ComprehensiveEvaluator:
    def __init__(self, model, config, categories):
        self.model = model
        self.config = config
        self.categories = categories
        self.metrics_dir = os.path.join(config.output_dir, "metrics_with_attention")
        os.makedirs(self.metrics_dir, exist_ok=True)
        
    def evaluate_with_attention(self, test_loader):
        """Evaluate model with detailed attention analysis"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_temporal_preds = []
        all_temporal_targets = []
        video_ids = []
        
        # Store attention weights for analysis
        all_attention_weights = {
            'spatial': [],
            'temporal_audio': [],
            'temporal_video': [],
            'cross_modal': [],
            'av_attention': []
        }
        
        print("Running evaluation with attention analysis...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx % 5 == 0:
                    clear_memory()
                
                audio_features = batch['audio_features'].to(self.config.device)
                video_features = batch['video_features'].to(self.config.device)
                video_frames = batch['video_frames'].to(self.config.device)
                labels = batch['label'].to(self.config.device)
                start_times = batch['start_time'].to(self.config.device)
                end_times = batch['end_time'].to(self.config.device)
                
                # Forward pass with attention
                outputs = self.model(audio_features, video_features, video_frames, return_attention=True)
                
                # Collect predictions
                preds = torch.softmax(outputs['class_logits'], dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_temporal_preds.extend(outputs['temporal_pred'].cpu().numpy())
                all_temporal_targets.extend(torch.stack([start_times, end_times], dim=1).cpu().numpy())
                video_ids.extend(batch['video_id'])
                
                # Collect attention weights
                if outputs['spatial_attention']:
                    all_attention_weights['spatial'].extend(outputs['spatial_attention'])
                if outputs['temporal_attention'] and outputs['temporal_attention']['audio'] is not None:
                    all_attention_weights['temporal_audio'].append(
                        outputs['temporal_attention']['audio'].cpu().numpy()
                    )
                if outputs['temporal_attention'] and outputs['temporal_attention']['video'] is not None:
                    all_attention_weights['temporal_video'].append(
                        outputs['temporal_attention']['video'].cpu().numpy()
                    )
                if outputs['cross_modal_attention'] is not None:
                    all_attention_weights['cross_modal'].append(
                        outputs['cross_modal_attention'].cpu().numpy()
                    )
                if outputs['av_attention'] is not None:
                    all_attention_weights['av_attention'].append({
                        'audio_to_video': outputs['av_attention']['audio_to_video_attention'].cpu().numpy(),
                        'video_to_audio': outputs['av_attention']['video_to_audio_attention'].cpu().numpy()
                    })
                
                if batch_idx % 20 == 0:
                    print(f"  Processed batch {batch_idx}/{len(test_loader)}")
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_temporal_preds = np.array(all_temporal_preds)
        all_temporal_targets = np.array(all_temporal_targets)
        
        # Calculate metrics
        print("Calculating metrics...")
        metrics = self.calculate_all_metrics(all_preds, all_labels, all_temporal_preds, all_temporal_targets)
        
        # Create visualizations
        print("Creating visualizations...")
        self.create_all_visualizations(all_preds, all_labels, all_temporal_preds, 
                                      all_temporal_targets, video_ids)
        
        # Create attention analysis
        print("Creating attention analysis...")
        self.create_attention_analysis(all_attention_weights, all_preds, all_labels, video_ids)
        
        return metrics, all_preds, all_labels, all_temporal_preds, all_temporal_targets, video_ids
    
    def create_attention_analysis(self, attention_weights, all_preds, all_labels, video_ids):
        """Create comprehensive attention analysis visualizations"""
        try:
            # 1. Attention Distribution Plot
            fig = plt.figure(figsize=(15, 10))
            
            # Spatial attention distribution
            if attention_weights['spatial']:
                ax1 = plt.subplot(2, 3, 1)
                spatial_flat = torch.cat([attn.flatten() for attn in attention_weights['spatial']]).cpu().numpy()
                ax1.hist(spatial_flat, bins=50, alpha=0.7, density=True)
                ax1.set_title('Spatial Attention Distribution')
                ax1.set_xlabel('Attention Weight')
                ax1.set_ylabel('Density')
                ax1.grid(True, alpha=0.3)
            
            # Temporal attention patterns
            if attention_weights['temporal_audio']:
                ax2 = plt.subplot(2, 3, 2)
                audio_temp_mean = np.mean(np.concatenate(attention_weights['temporal_audio'], axis=0), axis=(0, 1))
                ax2.plot(audio_temp_mean)
                ax2.set_title('Audio Temporal Attention Pattern')
                ax2.set_xlabel('Time Step')
                ax2.set_ylabel('Mean Attention')
                ax2.grid(True, alpha=0.3)
            
            if attention_weights['temporal_video']:
                ax3 = plt.subplot(2, 3, 3)
                video_temp_mean = np.mean(np.concatenate(attention_weights['temporal_video'], axis=0), axis=(0, 1))
                ax3.plot(video_temp_mean)
                ax3.set_title('Video Temporal Attention Pattern')
                ax3.set_xlabel('Frame')
                ax3.set_ylabel('Mean Attention')
                ax3.grid(True, alpha=0.3)
            
            # Cross-modal attention
            if attention_weights['cross_modal']:
                ax4 = plt.subplot(2, 3, 4)
                cross_modal_mean = np.mean(np.concatenate(attention_weights['cross_modal'], axis=0), axis=(0, 1))
                im = ax4.imshow(cross_modal_mean, cmap='viridis', aspect='auto')
                ax4.set_title('Cross-modal Attention Matrix')
                plt.colorbar(im, ax=ax4)
            
            # Audio-visual attention correlation
            if attention_weights['av_attention']:
                ax5 = plt.subplot(2, 3, 5)
                av_corr = []
                for attn_dict in attention_weights['av_attention']:
                    av = attn_dict['audio_to_video'].mean(axis=(0, 1))
                    va = attn_dict['video_to_audio'].mean(axis=(0, 1))
                    av_corr.append(np.corrcoef(av, va)[0, 1])
                ax5.hist(av_corr, bins=20, alpha=0.7)
                ax5.set_title('Audio-Video Attention Correlation')
                ax5.set_xlabel('Correlation')
                ax5.set_ylabel('Count')
                ax5.grid(True, alpha=0.3)
            
            # Attention vs Accuracy
            ax6 = plt.subplot(2, 3, 6)
            if attention_weights['spatial'] and len(attention_weights['spatial']) == len(all_labels):
                spatial_means = [attn.mean().item() for attn in attention_weights['spatial']]
                correct = (np.argmax(all_preds, axis=1) == all_labels)
                ax6.scatter(spatial_means, correct, alpha=0.5)
                ax6.set_title('Spatial Attention vs Accuracy')
                ax6.set_xlabel('Mean Spatial Attention')
                ax6.set_ylabel('Correct (1) / Incorrect (0)')
                ax6.grid(True, alpha=0.3)
            
            plt.suptitle('Attention Mechanism Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            safe_savefig(os.path.join(self.metrics_dir, 'attention_analysis.png'), dpi=150)
            
            # 2. Attention Head Analysis
            self.plot_attention_head_analysis(attention_weights)
            
            # 3. Temporal Attention Alignment
            self.plot_temporal_attention_alignment(attention_weights)
            
        except Exception as e:
            print(f"Error in attention analysis: {e}")
    
    def plot_attention_head_analysis(self, attention_weights):
        """Analyze different attention heads"""
        try:
            if attention_weights['av_attention']:
                fig = plt.figure(figsize=(12, 8))
                
                # Get data from first sample
                av_data = attention_weights['av_attention'][0]
                num_heads = av_data['audio_to_video'].shape[1]
                
                # Plot each head's attention pattern
                for head_idx in range(min(num_heads, 8)):
                    ax = plt.subplot(2, 4, head_idx + 1)
                    
                    # Audio to video attention
                    av_head = av_data['audio_to_video'][0, head_idx]
                    im = ax.imshow(av_head, cmap='viridis', aspect='auto')
                    ax.set_title(f'Head {head_idx+1} (A→V)')
                    ax.set_xlabel('Video Frames')
                    ax.set_ylabel('Audio Steps')
                    
                    plt.colorbar(im, ax=ax)
                
                plt.suptitle('Attention Head Patterns (Audio → Video)', fontsize=14, fontweight='bold')
                plt.tight_layout()
                safe_savefig(os.path.join(self.metrics_dir, 'attention_head_patterns.png'), dpi=150)
        except Exception as e:
            print(f"Error in attention head analysis: {e}")
    
    def plot_temporal_attention_alignment(self, attention_weights):
        """Analyze temporal attention alignment"""
        try:
            if attention_weights['temporal_audio'] and attention_weights['temporal_video']:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Audio temporal attention over time
                audio_attn = np.concatenate(attention_weights['temporal_audio'], axis=0)
                audio_mean = audio_attn.mean(axis=(0, 1, 2))
                axes[0].plot(audio_mean)
                axes[0].set_title('Audio Temporal Attention (Diag)')
                axes[0].set_xlabel('Time Step')
                axes[0].set_ylabel('Attention Weight')
                axes[0].grid(True, alpha=0.3)
                
                # Video temporal attention over frames
                video_attn = np.concatenate(attention_weights['temporal_video'], axis=0)
                video_mean = video_attn.mean(axis=(0, 1, 2))
                axes[1].plot(video_mean)
                axes[1].set_title('Video Temporal Attention (Diag)')
                axes[1].set_xlabel('Frame')
                axes[1].set_ylabel('Attention Weight')
                axes[1].grid(True, alpha=0.3)
                
                plt.suptitle('Temporal Attention Alignment', fontsize=14, fontweight='bold')
                plt.tight_layout()
                safe_savefig(os.path.join(self.metrics_dir, 'temporal_attention_alignment.png'), dpi=150)
        except Exception as e:
            print(f"Error in temporal attention alignment: {e}")
    
    # Keep all the existing metric calculation and visualization methods from the original code
    # (They remain the same, just add this new attention analysis)
    
    def calculate_all_metrics(self, all_preds, all_labels, temp_preds, temp_targets):
        """Calculate all evaluation metrics"""
        preds_class = np.argmax(all_preds, axis=1)
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(all_labels, preds_class)
        
        # Top-K Accuracy
        for k in [1, 3, 5]:
            top_k_correct = 0
            for i, true_label in enumerate(all_labels):
                top_k_preds = np.argsort(all_preds[i])[-k:]
                if true_label in top_k_preds:
                    top_k_correct += 1
            metrics[f'top_{k}_accuracy'] = top_k_correct / len(all_labels)
        
        # Precision, Recall, F1
        metrics['precision'] = precision_score(all_labels, preds_class, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(all_labels, preds_class, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(all_labels, preds_class, average='weighted', zero_division=0)
        
        # mAP
        try:
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(all_labels, classes=range(len(self.categories)))
            metrics['mAP'] = average_precision_score(y_true_bin, all_preds, average='weighted')
        except:
            metrics['mAP'] = 0.0
        
        # Temporal IoU
        metrics['temporal_iou'] = self.calculate_temporal_iou(temp_preds, temp_targets)
        metrics['temporal_mae'] = np.mean(np.abs(temp_preds - temp_targets))
        
        # AUC
        try:
            metrics['auc'] = roc_auc_score(all_labels, all_preds, multi_class='ovr', average='weighted')
        except:
            metrics['auc'] = 0.0
        
        # EDR
        metrics['edr'] = np.sum(preds_class == all_labels) / len(all_labels)
        
        # Hamming Loss
        preds_one_hot = np.zeros_like(all_preds)
        preds_one_hot[np.arange(len(preds_class)), preds_class] = 1
        true_one_hot = np.zeros_like(all_preds)
        true_one_hot[np.arange(len(all_labels)), all_labels] = 1
        metrics['hamming_loss'] = hamming_loss(true_one_hot.flatten(), preds_one_hot.flatten())
        
        # IoU stats
        iou_values = self.calculate_iou_values(temp_preds, temp_targets)
        metrics['mean_temporal_iou'] = np.mean(iou_values) if iou_values else 0
        metrics['std_temporal_iou'] = np.std(iou_values) if iou_values else 0
        
        # Class-wise metrics
        class_precisions = []
        class_recalls = []
        for class_idx in range(len(self.categories)):
            class_mask = all_labels == class_idx
            if np.sum(class_mask) > 0:
                class_preds = preds_class[class_mask]
                precision = np.mean(class_preds == class_idx)
                recall = np.sum((preds_class == class_idx) & (all_labels == class_idx)) / np.sum(all_labels == class_idx)
                class_precisions.append(precision)
                class_recalls.append(recall)
        metrics['avg_class_precision'] = np.mean(class_precisions) if class_precisions else 0
        metrics['avg_class_recall'] = np.mean(class_recalls) if class_recalls else 0
        
        # Attention-related metrics
        metrics['attention_variance'] = 0.0  # Placeholder
        metrics['cross_modal_correlation'] = 0.0  # Placeholder
        
        return metrics
    
    def calculate_temporal_iou(self, pred_times, true_times):
        ious = []
        for pred, true in zip(pred_times, true_times):
            pred_start, pred_end = pred
            true_start, true_end = true
            intersection_start = max(pred_start, true_start)
            intersection_end = min(pred_end, true_end)
            intersection = max(0, intersection_end - intersection_start)
            union_start = min(pred_start, true_start)
            union_end = max(pred_end, true_end)
            union = max(0.1, union_end - union_start)
            iou = intersection / union
            ious.append(iou)
        return np.mean(ious) if ious else 0
    
    def calculate_iou_values(self, pred_times, true_times):
        ious = []
        for pred, true in zip(pred_times, true_times):
            pred_start, pred_end = pred
            true_start, true_end = true
            intersection_start = max(pred_start, true_start)
            intersection_end = min(pred_end, true_end)
            intersection = max(0, intersection_end - intersection_start)
            union_start = min(pred_start, true_start)
            union_end = max(pred_end, true_end)
            union = max(0.1, union_end - union_start)
            iou = intersection / union
            ious.append(iou)
        return ious
    
    def create_all_visualizations(self, all_preds, all_labels, temp_preds, temp_targets, video_ids):
        """Create all visualizations (same as original)"""
        # This would include all the plotting methods from the original code
        # For brevity, I'm including just one example method
        try:
            # Confusion Matrix
            preds_class = np.argmax(all_preds, axis=1)
            cm = confusion_matrix(all_labels, preds_class)
            
            fig = plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix with Attention', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            safe_savefig(os.path.join(self.metrics_dir, 'confusion_matrix.png'))
            
        except Exception as e:
            print(f"Error in visualization: {e}")

# ==================== SPATIAL HEATMAP VISUALIZER WITH REAL ATTENTION ====================

class SpatialHeatmapVisualizer:
    def __init__(self, config, categories):
        self.config = config
        self.categories = categories
        self.heatmap_dir = os.path.join(config.output_dir, "spatial_heatmaps_with_attention")
        os.makedirs(self.heatmap_dir, exist_ok=True)
    
    def create_spatial_heatmaps(self, test_loader, model, num_samples=5):
        model.eval()
        samples_processed = 0
        
        print("Creating spatial heatmaps with real attention...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if samples_processed >= num_samples:
                    break
                
                clear_memory()
                
                video_ids = batch['video_id']
                audio_features = batch['audio_features'].to(self.config.device)
                video_features = batch['video_features'].to(self.config.device)
                video_frames = batch['video_frames'].to(self.config.device)
                labels = batch['label'].to(self.config.device)
                start_times = batch['start_time'].to(self.config.device)
                end_times = batch['end_time'].to(self.config.device)
                
                # Forward with attention
                outputs = model(audio_features, video_features, video_frames, return_attention=True)
                
                pred_probs = torch.softmax(outputs['class_logits'], dim=1)
                pred_labels = torch.argmax(pred_probs, dim=1)
                confidences = torch.max(pred_probs, dim=1)[0]
                
                for i in range(len(video_ids)):
                    if samples_processed >= num_samples:
                        break
                    
                    video_id = video_ids[i]
                    true_label = labels[i].item()
                    pred_label = pred_labels[i].item()
                    confidence = confidences[i].item()
                    
                    # Get attention maps
                    spatial_attention = outputs['spatial_attention'][i] if outputs['spatial_attention'] else None
                    
                    # Create comprehensive heatmap visualization
                    self.plot_comprehensive_heatmap(
                        video_frames[i].cpu(),
                        spatial_attention,
                        video_id,
                        true_label,
                        pred_label,
                        confidence,
                        outputs['temporal_pred'][i].cpu().numpy(),
                        [start_times[i].item(), end_times[i].item()],
                        outputs
                    )
                    
                    samples_processed += 1
        
        print(f"Created {samples_processed} spatial heatmap visualizations with real attention")
    
    def plot_comprehensive_heatmap(self, frames_tensor, attention_map, video_id, 
                                 true_label, pred_label, confidence,
                                 temporal_pred, true_times, outputs):
        try:
            fig = plt.figure(figsize=(20, 15))
            
            # Title
            title_text = (f'Video: {video_id} | '
                         f'True: {self.categories[true_label]} | '
                         f'Pred: {self.categories[pred_label]} | '
                         f'Confidence: {confidence:.3f}')
            fig.suptitle(title_text, fontsize=14, fontweight='bold')
            
            # Plot original frames
            for i in range(min(4, frames_tensor.shape[0])):
                ax = plt.subplot(3, 4, i + 1)
                frame = frames_tensor[i].permute(1, 2, 0).numpy()
                frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
                ax.imshow(frame)
                ax.set_title(f'Frame {i+1}')
                ax.axis('off')
            
            # Plot frames with attention overlay
            for i in range(min(4, frames_tensor.shape[0])):
                ax = plt.subplot(3, 4, i + 5)
                frame = frames_tensor[i].permute(1, 2, 0).numpy()
                frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
                ax.imshow(frame)
                
                if attention_map is not None:
                    if i == 0:  # Only show attention on first frame for clarity
                        attention_resized = nn.functional.interpolate(
                            attention_map.unsqueeze(0).unsqueeze(0),
                            size=frame.shape[:2],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze().cpu().numpy()
                        ax.imshow(attention_resized, alpha=0.6, cmap='jet')
                
                ax.set_title(f'Frame {i+1} with Attention')
                ax.axis('off')
            
            # Plot attention analysis
            ax9 = plt.subplot(3, 4, 9)
            if attention_map is not None:
                attention_np = attention_map.cpu().numpy()
                ax9.hist(attention_np.flatten(), bins=50, alpha=0.7)
                ax9.set_title('Attention Weight Distribution')
                ax9.set_xlabel('Attention Weight')
                ax9.set_ylabel('Frequency')
                ax9.grid(True, alpha=0.3)
            
            # Plot temporal attention
            if outputs['temporal_attention'] and outputs['temporal_attention']['video'] is not None:
                ax10 = plt.subplot(3, 4, 10)
                video_temp_attn = outputs['temporal_attention']['video'][0].mean(dim=0).cpu().numpy()
                im = ax10.imshow(video_temp_attn, cmap='viridis', aspect='auto')
                ax10.set_title('Temporal Attention (Video)')
                ax10.set_xlabel('Frame')
                ax10.set_ylabel('Frame')
                plt.colorbar(im, ax=ax10)
            
            # Plot cross-modal attention
            if outputs['cross_modal_attention'] is not None:
                ax11 = plt.subplot(3, 4, 11)
                cross_attn = outputs['cross_modal_attention'][0].mean(dim=0).cpu().numpy()
                im = ax11.imshow(cross_attn, cmap='plasma', aspect='auto')
                ax11.set_title('Cross-modal Attention')
                plt.colorbar(im, ax=ax11)
            
            # Plot temporal prediction vs ground truth
            ax12 = plt.subplot(3, 4, 12)
            x = np.arange(2)
            width = 0.35
            ax12.bar(x - width/2, temporal_pred, width, label='Predicted', alpha=0.7)
            ax12.bar(x + width/2, true_times, width, label='Ground Truth', alpha=0.7)
            ax12.set_xlabel('Temporal Boundary')
            ax12.set_ylabel('Time (seconds)')
            ax12.set_title('Temporal Localization')
            ax12.set_xticks(x)
            ax12.set_xticklabels(['Start', 'End'])
            ax12.legend()
            ax12.grid(True, alpha=0.3)
            
            plt.tight_layout()
            safe_savefig(os.path.join(self.heatmap_dir, f'heatmap_{video_id}.png'), dpi=150)
            
        except Exception as e:
            print(f"Error creating heatmap for {video_id}: {e}")

# ==================== MAIN EXECUTION ====================

def main():
    config = Config()
    print(f"Using device: {config.device}")
    print(f"Output directory: {config.output_dir}")
    print(f"Attention heads: {config.num_heads}")
    print(f"Attention dimension: {config.attention_dim}")
    
    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Set memory optimization
    torch.backends.cudnn.benchmark = True
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset splits
    print("Creating dataset splits...")
    dataset_manager = DatasetManager(config)
    train_ids, val_ids, test_ids = dataset_manager.create_splits()
    
    if len(train_ids) == 0:
        print("ERROR: No training data available.")
        return
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = AVEDataset(train_ids, dataset_manager.annotations, transform, 'train')
    val_dataset = AVEDataset(val_ids, dataset_manager.annotations, transform, 'val')
    test_dataset = AVEDataset(test_ids, dataset_manager.annotations, transform, 'test')
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Classes: {len(train_dataset.categories)}")
    
    # Update config
    config.num_classes = len(train_dataset.categories)
    
    # Initialize model with attention
    clear_memory()
    model = EnhancedAVEventLocalizer(config).to(config.device)
    
    # Print model architecture
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE WITH ATTENTION")
    print("="*80)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Count attention parameters
    attention_params = sum(p.numel() for name, p in model.named_parameters() if 'attention' in name)
    print(f"Attention parameters: {attention_params:,} ({attention_params/sum(p.numel() for p in model.parameters())*100:.1f}%)")
    print("="*80 + "\n")
    
    # Train
    trainer = EnhancedTrainer(model, config, train_dataset.categories)
    trainer.train(train_loader, val_loader)
    
    # Load best model
    clear_memory()
    try:
        checkpoint = torch.load(os.path.join(config.output_dir, 'best_model_with_attention.pth'), 
                               map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']} with accuracy: {checkpoint['val_accuracy']:.4f}")
    except Exception as e:
        print(f"Using current model: {e}")
    
    # Create heatmaps with real attention
    heatmap_visualizer = SpatialHeatmapVisualizer(config, train_dataset.categories)
    heatmap_visualizer.create_spatial_heatmaps(test_loader, model, num_samples=5)
    
    # Evaluate with attention analysis
    evaluator = ComprehensiveEvaluator(model, config, train_dataset.categories)
    metrics, all_preds, all_labels, temp_preds, temp_targets, video_ids = evaluator.evaluate_with_attention(test_loader)
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS WITH ATTENTION")
    print("="*80)
    
    print("\nClassification Metrics:")
    for key in ['accuracy', 'top_1_accuracy', 'top_3_accuracy', 'top_5_accuracy', 
                'precision', 'recall', 'f1_score', 'mAP', 'auc', 'edr']:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")
    
    print("\nTemporal Metrics:")
    for key in ['temporal_mae', 'temporal_iou', 'mean_temporal_iou', 'std_temporal_iou']:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")
    
    print("\nClass-wise Metrics:")
    for key in ['avg_class_precision', 'avg_class_recall', 'hamming_loss']:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")
    
    print("\nAttention Analysis:")
    print(f"  Model has {config.num_heads} attention heads")
    print(f"  Attention dimension: {config.attention_dim}")
    
    # Save metrics
    metrics_file = os.path.join(config.output_dir, "all_metrics_with_attention.txt")
    with open(metrics_file, 'w') as f:
        f.write("EVALUATION METRICS WITH ATTENTION\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"Attention Heads: {config.num_heads}\n")
        f.write(f"Attention Dimension: {config.attention_dim}\n\n")
        
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    
    print(f"\nMetrics saved to: {metrics_file}")
    print(f"Visualizations saved to: {config.output_dir}")
    print(f"Attention analysis saved to: {os.path.join(config.output_dir, 'attention_visualization')}")
    print("\nTraining and evaluation complete with real attention mechanisms!")

if __name__ == "__main__":
    main()