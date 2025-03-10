import numpy as np
import matplotlib.pyplot as plt
import cv2
import functions

shading = 'g'

# Load data from .npy file
data_dict = np.load('hw1.npy', allow_pickle=True).item()

# Extract the data
vertices = data_dict['vertices']
vcolors = data_dict['vcolors']
faces = data_dict['faces']
depth = data_dict['depth']

# Render the image
image = functions.render_img(faces,vertices,vcolors,depth,shading)

# Since opencv library uses BGR format, we need to convert the image RGB to BGR
# Convert the image to a compatible data type (uint8) before converting to BGR
image_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)

# Convert RGB to BGR
image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)

# ! In order the image to be saved, the path shouldnt have greek characters
cv2.imwrite('gouraud_shading.jpg', image_bgr)

# Display the image using Matplotlib
plt.imshow(image)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()