import numpy as np
import cv2
import torch
from scipy.ndimage import center_of_mass

def preprocess_drawn_image(image_path_or_array):
    """
    Transforms raw hand-drawn images into a centered 1x1x28x28 normalized PyTorch tensor.
    """
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array, cv2.IMREAD_GRAYSCALE)
    else:
        img = image_path_or_array.copy()

    # Invert colors: MNIST expects white strokes on black backgrounds
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)

    # Threshold to remove stroke feathering and faint noise
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    # Find the bounding box around the digit
    coords = cv2.findNonZero(img)
    if coords is None:
        return torch.zeros((1, 1, 28, 28), dtype=torch.float32)

    x, y, w, h = cv2.boundingRect(coords)
    digit_crop = img[y:y+h, x:x+w]

    # Resize bounding box to fit within 20x20 window preserving aspect ratio
    if w > h:
        new_w = 20
        new_h = max(1, int(round((h * 20.0 / w))))
    else:
        new_h = 20
        new_w = max(1, int(round((w * 20.0 / h))))
        
    digit_resized = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Place resized digit into a 28x28 black canvas
    padded = np.zeros((28, 28), dtype=np.float32)
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    padded[start_y:start_y + new_h, start_x:start_x + new_w] = digit_resized

    # Center by Center of Mass (critical for MNIST alignment)
    cy, cx = center_of_mass(padded)
    shift_x = np.round(14.0 - cx)
    shift_y = np.round(14.0 - cy)

    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    centered = cv2.warpAffine(padded, M, (28, 28))

    # Normalize with MNIST training mean and standard deviation
    norm_img = (centered / 255.0 - 0.1307) / 0.3081

    # Reshape to [1, 1, 28, 28] tensor
    return torch.from_numpy(norm_img).float().unsqueeze(0).unsqueeze(0)