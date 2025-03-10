import numpy as np
import matplotlib.pyplot as plt
import cv2
import tmap as map

data_dict = np.load('h3.npy', allow_pickle=True).item()

#Extract the data
verts = data_dict['verts']
vert_colors = data_dict['vertex_colors'].T
faces = data_dict['face_indices']
uvs = data_dict['uvs']
face_uvs = data_dict['face_uv_indices'].T
eye = data_dict['cam_eye']
up = data_dict['cam_up']
target = data_dict['cam_lookat']
ka = data_dict['ka']
kd = data_dict['kd']
ks = data_dict['ks']
n = data_dict['n']
light_positions = data_dict['light_positions']
light_intensities = data_dict['light_intensities']
Ia = data_dict['Ia'].T
M = data_dict['M']
N = data_dict['N']
W = data_dict['W']
H = data_dict['H']
bg_color = data_dict['bg_color'].T
focal = data_dict['focal']

# The image is now a MxNx3 numpy array but the color values are between 0 and 255
text_map = cv2.imread('cat_diff.png')
text_map = cv2.cvtColor(text_map, cv2.COLOR_BGR2RGB)
# The image is now a MxNx3 numpy array but the values are between 0 and 255
# Normalize the values to be between 0 and 1
text_map = text_map / 255.0

# Image with Gouraud shading and lighting
img = map.render_object_map("gouraud", focal, eye, target, up, bg_color, M, N, H, W,
                    verts, vert_colors, faces, ka, kd, ks, n, light_positions, light_intensities, Ia, uvs, face_uvs, text_map)
plt.imsave('mapGouraudLight.jpg', np.array(img[::-1]))

# Image with Phong shading and lighting
img = map.render_object_map("phong", focal, eye, target, up, bg_color, M, N, H, W,
                    verts, vert_colors, faces, ka, kd, ks, n, light_positions, light_intensities, Ia, uvs, face_uvs, text_map)
plt.imsave('mapPhongLight.jpg', np.array(img[::-1]))

# Image with no lighting and gouard shading
img = map.render_object_map("nolight", focal, eye, target, up, bg_color, M, N, H, W,
                    verts, vert_colors, faces, ka, kd, ks, n, light_positions, light_intensities, Ia, uvs, face_uvs, text_map)
plt.imsave('mapNoLight.jpg', np.array(img[::-1]))