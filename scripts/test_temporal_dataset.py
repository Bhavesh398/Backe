from backend.video.temporal_dataset import VideoTemporalDataset

dataset = VideoTemporalDataset(
    root_dir="datasets/videos",
    frames_per_clip=16
)

clip, label = dataset[0]

print("Clip shape:", clip.shape)
print("Label:", label)
