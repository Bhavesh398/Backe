"""
Test script to diagnose model issues
"""
import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.image_model import DeepfakeImageModel
from backend.preprocessing.image import load_image

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = DeepfakeImageModel()
    MODEL_PATH = "models/image_deepfake.pth"
    
    print(f"Loading model from: {MODEL_PATH}")
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        
        # Check if checkpoint contains extra info
        if isinstance(checkpoint, dict):
            print("\n=== Checkpoint Contents ===")
            print(f"Keys: {checkpoint.keys()}")
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
                print(f"Training Accuracy: {checkpoint.get('accuracy', 'N/A')}")
                print(f"Training Loss: {checkpoint.get('loss', 'N/A')}")
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.to(device)
    model.eval()
    
    # Print model architecture
    print("\n=== Model Architecture ===")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test with a dummy image
    print("\n=== Testing with random input ===")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
        probs = torch.softmax(output, dim=1)[0]
        
    print(f"Raw output: {output}")
    print(f"Softmax probabilities: {probs}")
    print(f"Real probability: {probs[0].item():.4f}")
    print(f"Fake probability: {probs[1].item():.4f}")
    
    # Test if model always outputs same values
    print("\n=== Testing for stuck weights ===")
    outputs = []
    for i in range(5):
        dummy = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            out = model(dummy)
            probs = torch.softmax(out, dim=1)[0]
            outputs.append(probs[1].item())
    
    print(f"5 random inputs fake probs: {outputs}")
    
    if max(outputs) - min(outputs) < 0.01:
        print("⚠️  WARNING: Model outputs are nearly identical!")
        print("   This suggests the model is not trained or weights are stuck.")
    
    # Check weight statistics
    print("\n=== Weight Statistics ===")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}")
            if param.data.std() < 0.001:
                print(f"  ⚠️  WARNING: Very low std deviation - weights may not be trained!")

if __name__ == "__main__":
    test_model()
