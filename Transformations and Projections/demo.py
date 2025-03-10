import numpy as np
import matplotlib.pyplot as plt
import cv2
import transform as t

data_dict = np.load('hw2.npy', allow_pickle=True).item()

#Extract the data
v_pos = data_dict['v_pos']
v_clr = data_dict['v_clr']
t_pos_idx = data_dict['t_pos_idx']
eye = data_dict['eye']
up = data_dict['up']
target = data_dict['target']
focal = data_dict['focal']
plane_h = data_dict['plane_h']
plane_w = data_dict['plane_w']
res_h = data_dict['res_h']
res_w = data_dict['res_w']
theta_0 = data_dict['theta_0']
rot_axis_0 = data_dict['rot_axis_0']
t_0 = data_dict['t_0']
t_1 = data_dict['t_1']

# 0. STARTING POSITION
image = t.render_object(v_pos.T, v_clr, t_pos_idx, plane_h, plane_w, res_h, res_w, focal, eye, up, target)
# The render_object function takes Nx3 v_pos array as input so we need to transpose the v_pos array

# Since opencv library uses BGR format, we need to convert the image RGB to BGR
# Convert the image to a compatible data type (uint8) before converting to BGR
image_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
# Convert RGB to BGR
image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
# Save the image (corrected conversion to BGR)
cv2.imwrite('0.jpg', image_bgr)


# STARTS THE TRANSFORMATIONS - the input of each transformation is the output of the previous transformation
transform_obj = t.Transform()

# 1. Rotate the object by theta_0 around the axis rot_axis_0
transform_obj.rotate(theta_0, rot_axis_0) # Sets the rotation matrix
v_pos_rotated = transform_obj.transform_pts(v_pos.T) # Transform the points - transform_pts takes Nx3 array as argument

image = t.render_object(v_pos_rotated.T, v_clr, t_pos_idx, plane_h, plane_w, res_h, res_w, focal, eye, up, target)

# Since opencv library uses BGR format, we need to convert the image RGB to BGR
# Convert the image to a compatible data type (uint8) before converting to BGR
image_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
# Convert RGB to BGR
image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
# Save the image (corrected conversion to BGR)
cv2.imwrite('1.jpg', image_bgr)


# 2. Translate the object by t_0
transform_obj.__init__() # Resets the transformation matrix
transform_obj.translate(t_0) # Sets the translation matrix
v_pos_rot_trans1 = transform_obj.transform_pts(v_pos_rotated.T) # Transform the points

image = t.render_object(v_pos_rot_trans1.T, v_clr, t_pos_idx, plane_h, plane_w, res_h, res_w, focal, eye, up, target)

# Same as before
image_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
cv2.imwrite('2.jpg', image_bgr)


# 3. Translate the object by t_1
transform_obj.__init__() # Resets again the transformation matrix
transform_obj.translate(t_1)
v_pos_rot_trans1_trans2 = transform_obj.transform_pts(v_pos_rot_trans1.T)

image = t.render_object(v_pos_rot_trans1_trans2.T, v_clr, t_pos_idx, plane_h, plane_w, res_h, res_w, focal, eye, up, target)

image_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
cv2.imwrite('3.jpg', image_bgr)