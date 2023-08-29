import numpy as np
from shipping_prediction.utils import load_numpy_array_data()
# Load the .npz file
data = np.load(r'F:\Shipping_Pricing_Prediction\artifact\08282023__160814\data_transformation\transformed\train.npz')

# Print the keys in the .npz file
print("Keys in the .npz file:", data.files)

