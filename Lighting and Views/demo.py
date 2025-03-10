import numpy as np
import matplotlib.pyplot as plt
import cv2
import lighting as l
import render as r
import time

data_dict = np.load('h3.npy', allow_pickle=True).item()

#Extract the data
verts = data_dict['verts']
vert_colors = data_dict['vertex_colors'].T
faces = data_dict['face_indices']
# uvs = data_dict['uvs']
# face_uvs = data_dict['face_uv_indices'].T
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

start_time = time.time()

# GOURAUD
# ONLY AMBIENT LIGHT
# kd = 0 and ks = 0

img = r.render_object("gouraud", focal, eye, target, up, bg_color, M, N, H, W,
                    verts, vert_colors, faces, ka, 0, 0, n, light_positions, light_intensities, Ia)
plt.imsave('0.jpg', np.array(img[::-1]))

# ONLY DIFFUSE LIGHT
# ka = 0 and ks = 0
img = r.render_object("gouraud", focal, eye, target, up, bg_color, M, N, H, W,
                    verts, vert_colors, faces, 0, kd, 0, n, light_positions, light_intensities, Ia)
plt.imsave('1.jpg', np.array(img[::-1]))

# ONLY SPECULAR LIGHT
# ka = 0 and kd = 0
img = r.render_object("gouraud", focal, eye, target, up, bg_color, M, N, H, W,
                    verts, vert_colors, faces, 0, 0, ks, n, light_positions, light_intensities, Ia)
plt.imsave('2.jpg', np.array(img[::-1]))

# ALL LIGHTS
img = r.render_object("gouraud", focal, eye, target, up, bg_color, M, N, H, W,
                    verts, vert_colors, faces, ka, kd, ks, n, light_positions, light_intensities, Ia)
plt.imsave('3.jpg', np.array(img[::-1]))

print("Objects rendered in", time.time() - start_time, "sec")
start_time = time.time()

# PHONG
# ONLY AMBIENT LIGHT
# kd = 0 and ks = 0
img = r.render_object("phong", focal, eye, target, up, bg_color, M, N, H, W,
                    verts, vert_colors, faces, ka, 0, 0, n, light_positions, light_intensities, Ia)
plt.imsave('4.jpg', np.array(img[::-1]))

# ONLY DIFFUSE LIGHT
# ka = 0 and ks = 0
img = r.render_object("phong", focal, eye, target, up, bg_color, M, N, H, W,
                    verts, vert_colors, faces, 0, kd, 0, n, light_positions, light_intensities, Ia)
plt.imsave('5.jpg', np.array(img[::-1]))

# ONLY SPECULAR LIGHT
# ka = 0 and kd = 0

img = r.render_object("phong", focal, eye, target, up, bg_color, M, N, H, W,
                    verts, vert_colors, faces, 0, 0, ks, n, light_positions, light_intensities, Ia)
plt.imsave('6.jpg', np.array(img[::-1]))

# ALL LIGHTS
img = r.render_object("phong", focal, eye, target, up, bg_color, M, N, H, W,
                    verts, vert_colors, faces, ka, kd, ks, n, light_positions, light_intensities, Ia)
plt.imsave('7.jpg', np.array(img[::-1]))
print("Phong objects rendered in", time.time() - start_time, "sec")


# GOURAUD for every light source
for i in range(len(light_positions)):
    img = r.render_object("gouraud", focal, eye, target, up, bg_color, M, N, H, W,
                    verts, vert_colors, faces, ka, kd, ks, n, light_positions[i], light_intensities[i], Ia)
    plt.imsave('g_light' + str(i) + '.jpg', np.array(img[::-1]))

# PHONG for every light source
for i in range(len(light_positions)):
    img = r.render_object("phong", focal, eye, target, up, bg_color, M, N, H, W,
                    verts, vert_colors, faces, ka, kd, ks, n, light_positions[i], light_intensities[i], Ia)
    plt.imsave('p_light' + str(i) + '.jpg', np.array(img[::-1]))