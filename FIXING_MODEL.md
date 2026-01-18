# Fixing Your Deepfake Detection Model

## Problem Summary
Your model shows similar probabilities (~53-54%) for both real and fake images, indicating it's not properly distinguishing between them. This is typically caused by:

1. **Untrained or poorly trained model** - Most common cause
2. **Wrong preprocessing** - Mismatch between training and inference
3. **Class index confusion** - Real/Fake labels swapped
4. **Model architecture issues** - Frozen weights or gradient problems

## Quick Diagnosis

### Step 1: Test Your Current Model

Run the diagnostic script to check if your model is actually trained:

```powershell
cd d:\deepscan-truth-forge-main\projects\deepfake-forensics
python backend/test_model.py
```

**What to look for:**
- ✅ Model outputs should vary significantly for different inputs (range > 0.3)
- ❌ If outputs are nearly identical (range < 0.01), model weights are stuck/untrained
- Check weight statistics - std dev should be > 0.01 for trained weights

### Step 2: Check Your Training Data

Your training data should be organized as:
```
data/
  train/
    real/          <- Put real images here
      img1.jpg
      img2.jpg
      ...
    fake/          <- Put fake/AI-generated images here
      img1.jpg
      img2.jpg
      ...
```

**Requirements:**
- At least 100-200 images per class (more is better)
- Balanced dataset (roughly equal real and fake images)
- Diverse samples (different faces, angles, lighting, etc.)

### Step 3: Retrain the Model

If your model is not properly trained (diagnosis shows stuck weights):

```powershell
# Quick training (10 epochs)
python backend/fix_model.py --data_dir data/train --epochs 10

# Longer training for better results (recommended)
python backend/fix_model.py --data_dir data/train --epochs 30 --lr 0.0001

# With custom batch size
python backend/fix_model.py --data_dir data/train --epochs 30 --batch_size 32
```

**Expected Results:**
- Training accuracy should reach > 80% by epoch 10
- Validation accuracy should be > 70%
- Loss should steadily decrease

If accuracy stays near 50%, you need more/better training data.

## What Changed in app.py

### 1. Added Logging
Now logs raw model outputs so you can see what's happening:
```python
logger.info(f"Real prob: {real_prob:.4f}, Fake prob: {fake_prob:.4f}")
```

### 2. Fixed Thresholds
Changed from arbitrary thresholds (0.55, 0.70) to using 0.5 as the decision boundary:
- fake_prob > 0.5 → FAKE
- fake_prob < 0.5 → REAL

### 3. Added Both Probabilities
Now returns both real and fake probabilities for transparency:
```json
{
  "real_probability": 47.23,
  "fake_probability": 52.77,
  "verdict": "FAKE",
  "confidence": 5.54
}
```

### 4. Added Confidence Score
Shows how confident the model is (0-100 scale based on distance from 50%)

## Understanding the Output

When you upload an image, check the terminal logs:

```
INFO: Real prob (class 0): 0.4723, Fake prob (class 1): 0.5277
```

**Good Model:**
- Real images: fake_prob should be < 0.3 (high confidence REAL)
- Fake images: fake_prob should be > 0.7 (high confidence FAKE)

**Bad Model (your current issue):**
- Both real and fake: fake_prob around 0.5 (random guessing)
- This means the model is not trained!

## Common Issues & Solutions

### Issue 1: Model Always Returns ~50%
**Cause:** Model is not trained or weights didn't update
**Solution:** 
1. Run `test_model.py` to confirm
2. Retrain using `fix_model.py`
3. Ensure you have good training data

### Issue 2: Model Predicts Opposite (Real as Fake, Fake as Real)
**Cause:** Class labels are swapped in training
**Solution:** In app.py, swap the indices:
```python
# Change from:
fake_prob = probs[1].item()
# To:
fake_prob = probs[0].item()
```

### Issue 3: Low Accuracy Even After Training
**Cause:** Insufficient or poor quality training data
**Solution:**
- Get more diverse training images (aim for 500+ per class)
- Use data augmentation (rotation, flip, color jitter)
- Try a better pretrained model (ResNet50, EfficientNet)

### Issue 4: Good Training Accuracy but Bad Real-World Performance
**Cause:** Overfitting or domain mismatch
**Solution:**
- Add more validation data
- Use stronger data augmentation
- Test on samples from same distribution as real use-case

## Testing After Fixes

1. **Test with Known Samples:**
   - Upload a real photo you took with your phone
   - Upload an AI-generated image from a tool like Midjourney/DALL-E
   - They should show clearly different probabilities

2. **Check the Logs:**
   - Look at terminal output for probability values
   - Real images should consistently show fake_prob < 0.3
   - Fake images should consistently show fake_prob > 0.7

3. **Test Multiple Images:**
   - Test at least 5-10 real and 5-10 fake images
   - Calculate accuracy manually: (correct predictions / total) × 100
   - Should be > 80% if model is working

## Where to Get Training Data

### Real Images:
- FFHQ dataset (Flickr-Faces-HQ)
- CelebA dataset
- Your own photos

### Fake Images:
- FaceForensics++ dataset
- Deepfake Detection Challenge dataset
- Generate using StyleGAN, DALL-E, Midjourney, etc.

### Ready-to-Use Datasets:
- **FaceForensics++:** https://github.com/ondyari/FaceForensics
- **Celeb-DF:** http://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html
- **DFDC:** https://www.kaggle.com/c/deepfake-detection-challenge

## Next Steps

1. **Run Diagnostics:**
   ```powershell
   python backend/test_model.py
   ```

2. **If Model is Untrained:**
   ```powershell
   # Organize your data first, then:
   python backend/fix_model.py --epochs 30
   ```

3. **Test the API:**
   ```powershell
   # Start the server
   uvicorn backend.app:app --reload
   
   # Test with real and fake images
   # Check the terminal logs for probability values
   ```

4. **Iterate:**
   - If accuracy is still low, get more training data
   - Try training for more epochs (50-100)
   - Consider using a better backbone model

## Model Architecture Notes

If you're using a simple CNN, consider upgrading to:
- **ResNet18/50:** Better feature extraction
- **EfficientNet:** More efficient and accurate
- **Xception:** Specifically designed for deepfake detection

These require modifying `backend/models/image_model.py`.

## Need More Help?

Check these in order:
1. Model weights statistics (std dev should be > 0.01)
2. Training data quality and quantity
3. Preprocessing consistency between training and inference
4. Model architecture complexity vs. data amount

The key is: **If probabilities are near 50%, the model is essentially random - you must retrain it!**
