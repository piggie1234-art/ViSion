import json
import numpy as np

# Load the JSON file
with open('CameraParameters.json', 'r') as f:
    data = json.load(f)

# Convert lists from JSON file to numpy arrays
intrinsic_matrix1 = np.array(data['IntrinsicMatrix1'])
intrinsic_matrix2 = np.array(data['IntrinsicMatrix2'])
rotation_matrix = np.array(data['RotationMatrix'])
translation_vector = np.array(data['TranslationVector'])

print('Intrinsic Matrix Camera 1:\n', intrinsic_matrix1)
print('Intrinsic Matrix Camera 2:\n', intrinsic_matrix2)
print('Rotation Matrix:\n', rotation_matrix)
print('Translation Vector:\n', translation_vector)