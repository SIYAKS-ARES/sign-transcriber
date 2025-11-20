"""
Random Test Sample Inference
Test setinden rastgele örnekleri tahmin eder
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from models.transformer_model import TransformerSignLanguageClassifier
from config import TransformerConfig

# Initialize config
config = TransformerConfig()

# Load test data
print('📂 Loading test data...')
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')
print(f'   ✅ Test data: {X_test.shape}')

# Device
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('🖥️  Device: MPS (Apple Silicon GPU)')
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print('🖥️  Device: CUDA')
else:
    device = torch.device('cpu')
    print('🖥️  Device: CPU')

# Load best model
print('\n🏗️  Loading model...')
model = TransformerSignLanguageClassifier(
    input_dim=config.INPUT_DIM,
    num_classes=config.NUM_CLASSES,
    d_model=config.D_MODEL,
    nhead=config.NHEAD,
    num_encoder_layers=config.NUM_ENCODER_LAYERS,
    dim_feedforward=config.DIM_FEEDFORWARD,
    dropout=config.DROPOUT,
    pooling_type=config.POOLING_TYPE
).to(device)

checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print('   ✅ Model loaded!')

# Test 20 random samples
num_samples = 20
indices = np.random.choice(len(X_test), num_samples, replace=False)

print(f'\n🎯 Testing {num_samples} Random Samples:')
print('═' * 80)

correct = 0
predictions = []

for idx in indices:
    x = torch.FloatTensor(X_test[idx:idx+1]).to(device)
    with torch.no_grad():
        logits = model(x, mask=None)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        confidence = probs[0, pred].item()
    
    true_label = y_test[idx]
    true_class = config.CLASS_NAMES[true_label]
    pred_class = config.CLASS_NAMES[pred]
    
    is_correct = (pred == true_label)
    status = '✅' if is_correct else '❌'
    
    print(f'{status} True: {true_class:15s} | Pred: {pred_class:15s} | Conf: {confidence:.3f}')
    
    if is_correct:
        correct += 1
    
    predictions.append({
        'true': true_class,
        'pred': pred_class,
        'correct': is_correct,
        'confidence': confidence
    })

print('═' * 80)
print(f'📊 Accuracy on {num_samples} samples: {correct}/{num_samples} = {correct/num_samples*100:.1f}%')

# Confidence stats
confidences = [p['confidence'] for p in predictions]
correct_confidences = [p['confidence'] for p in predictions if p['correct']]
wrong_confidences = [p['confidence'] for p in predictions if not p['correct']]

print(f'\n📈 Confidence Statistics:')
print(f'   Overall:  Mean={np.mean(confidences):.3f}, Std={np.std(confidences):.3f}')
if correct_confidences:
    print(f'   Correct:  Mean={np.mean(correct_confidences):.3f}, Std={np.std(correct_confidences):.3f}')
if wrong_confidences:
    print(f'   Wrong:    Mean={np.mean(wrong_confidences):.3f}, Std={np.std(wrong_confidences):.3f}')

