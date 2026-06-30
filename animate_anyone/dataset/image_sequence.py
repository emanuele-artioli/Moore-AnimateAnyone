import json
import random
import glob
import os
from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor

def pad_to_square(img, fill=0):
    w, h = img.size
    max_dim = max(w, h)
    pad_left = (max_dim - w) // 2
    pad_top = (max_dim - h) // 2
    pad_right = max_dim - w - pad_left
    pad_bottom = max_dim - h - pad_top
    return F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=fill, padding_mode='constant')

class ImageSequenceDataset(Dataset):
    """
    A generic dataset that reads frame sequences from directories instead of MP4s, 
    avoiding video compression artifacts. Also supports optional letterbox padding 
    to preserve aspect ratios instead of RandomResizedCrop.
    """
    def __init__(
        self,
        sample_rate,
        n_sample_frames,
        width,
        height,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        drop_ratio=0.1,
        data_meta_paths=["./data/anyone_meta.json"],
        pad_to_square=False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.width = width
        self.height = height
        self.pad_to_square = pad_to_square
        
        # Original parameters for RandomResizedCrop (used if pad_to_square is False)
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.drop_ratio = drop_ratio

        vid_meta = []
        for data_meta_path in data_meta_paths:
            with open(data_meta_path, "r") as f:
                vid_meta.extend(json.load(f))
        self.vid_meta = vid_meta

        self.clip_image_processor = CLIPImageProcessor()

        if self.pad_to_square:
            self.pixel_transform = transforms.Compose([
                transforms.Resize((self.height, self.width), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor()
            ])
        else:
            self.pixel_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        (height, width),
                        scale=self.img_scale,
                        ratio=self.img_ratio,
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),
                    transforms.ToTensor(),
                ]
            )

        self.cond_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):
        video_meta = self.vid_meta[index]
        video_path = video_meta["video_path"]
        kps_path = video_meta["kps_path"]

        # Read frames from directory
        video_frames = sorted(glob.glob(os.path.join(video_path, "*.png")) + glob.glob(os.path.join(video_path, "*.jpg")))
        kps_frames = sorted(glob.glob(os.path.join(kps_path, "*.png")) + glob.glob(os.path.join(kps_path, "*.jpg")))
        
        video_length = len(video_frames)
        margin = min(10, video_length)

        ref_img_idx = random.randint(0, video_length - 1)
        if self.n_sample_frames > 1:
            if ref_img_idx + margin < video_length:
                tgt_img_idx = random.randint(ref_img_idx + margin, video_length - 1)
            elif ref_img_idx - margin > 0:
                tgt_img_idx = random.randint(0, ref_img_idx - margin)
            else:
                tgt_img_idx = random.randint(0, video_length - 1)
                
            # Sample sequence
            start_idx = max(0, tgt_img_idx - self.n_sample_frames * self.sample_rate)
            end_idx = min(video_length, start_idx + self.n_sample_frames * self.sample_rate)
            tgt_frame_indices = list(range(start_idx, end_idx, self.sample_rate))
            if len(tgt_frame_indices) < self.n_sample_frames:
                # Pad if too short
                tgt_frame_indices = tgt_frame_indices + [tgt_frame_indices[-1]] * (self.n_sample_frames - len(tgt_frame_indices))
        else:
            tgt_img_idx = random.randint(0, video_length - 1)
            tgt_frame_indices = [tgt_img_idx]

        ref_img_path = video_frames[ref_img_idx]
        ref_img = Image.open(ref_img_path)
        if ref_img.mode == 'RGBA':
            bg = Image.new('RGBA', ref_img.size, (0,0,0,255))
            ref_img = Image.alpha_composite(bg, ref_img).convert("RGB")
        else:
            ref_img = ref_img.convert("RGB")

        tgt_img_list = []
        tgt_pose_list = []
        for tgt_idx in tgt_frame_indices:
            tgt_img_path = video_frames[tgt_idx]
            tgt_img = Image.open(tgt_img_path)
            if tgt_img.mode == 'RGBA':
                bg = Image.new('RGBA', tgt_img.size, (0,0,0,255))
                tgt_img = Image.alpha_composite(bg, tgt_img).convert("RGB")
            else:
                tgt_img = tgt_img.convert("RGB")
            
            tgt_pose_path = kps_frames[tgt_idx]
            tgt_pose = Image.open(tgt_pose_path)
            if tgt_pose.mode == 'RGBA':
                bg = Image.new('RGBA', tgt_pose.size, (0,0,0,255))
                tgt_pose = Image.alpha_composite(bg, tgt_pose).convert("RGB")
            else:
                tgt_pose = tgt_pose.convert("RGB")
                
            tgt_img_list.append(tgt_img)
            tgt_pose_list.append(tgt_pose)

        if self.pad_to_square:
            ref_img = pad_to_square(ref_img, fill=0)
            tgt_img_list = [pad_to_square(img, fill=0) for img in tgt_img_list]
            tgt_pose_list = [pad_to_square(img, fill=0) for img in tgt_pose_list]

        state = torch.get_rng_state()
        tgt_img_tensor = self.augmentation(tgt_img_list, self.pixel_transform, state)
        tgt_pose_tensor = self.augmentation(tgt_pose_list, self.pixel_transform if self.pad_to_square else self.cond_transform, state)
        ref_img_tensor = self.augmentation(ref_img, self.pixel_transform, state)

        clip_ref_img_tensor = self.clip_image_processor(
            images=ref_img, return_tensors="pt"
        ).pixel_values[0]

        if self.n_sample_frames == 1:
            tgt_img_tensor = tgt_img_tensor[0]
            tgt_pose_tensor = tgt_pose_tensor[0]

        return {
            "dataset_name": "image_sequence",
            # Keys for stage 2
            "pixel_values_vid": tgt_img_tensor,
            "pixel_values_pose": tgt_pose_tensor,
            "clip_ref_img": clip_ref_img_tensor,
            "pixel_values_ref_img": ref_img_tensor,
            # Keys for stage 1
            "img": tgt_img_tensor,
            "tgt_pose": tgt_pose_tensor,
            "clip_images": clip_ref_img_tensor,
            "ref_img": ref_img_tensor,
        }

    def __len__(self):
        return len(self.vid_meta)
