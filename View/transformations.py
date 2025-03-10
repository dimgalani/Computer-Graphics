from typing import Tuple
import numpy as np
import functions as f

class Transform:
 # Interface for performing affine transformations.
    def __init__(self):
        # Initialize a Transform object.
        self.mat = np.eye(4) # Identity matrix
    def rotate(self, theta: float, u: np.ndarray)-> None:
        # rotate the transformation matrix
        self.mat[:3, :3] = np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * np.outer(u, u) + np.sin(theta) * np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]]) # Rodrigues' formula with homogeneous coordinates
    def translate(self, t: np.ndarray)-> None: #just updates the matrix
        # translate the transformation matrix.
        t = t.flatten() # Makes the 3x1 vector an one-dimensional array
        self.mat[:3, 3] += t # The last column of the three rows 
    def transform_pts(self, pts: np.ndarray)-> np.ndarray: #*input is a Nx3 array
        # transform the specified points according to our current matrix.
        pts = pts.T # Transpose the points to have a 3xN matrix
        pts_transformed = np.matmul(self.mat, np.vstack((pts, np.ones((1, pts.shape[1]))))) # Apply the transformation matrix to the points
        pts_transformed_without_homogeneous = pts_transformed[:-1]
        # The transformed points are in an array of size 3xN (since it isnt defined by the instructions)
        return pts_transformed_without_homogeneous

# OUTSIDE THE CLASS
def world2view(pts: np.ndarray, R: np.ndarray, c0: np.ndarray)-> np.ndarray:
        # Implements a world-to-view transform, i.e. transforms the specified points to the coordinate frame of a camera. The camera coordinate frame
        # is specified rotation (w.r.t. the world frame) and its point of reference (w.r.t. to the world frame).
        # pts: the points to transform 3xN - the columns contain the coordinates of the points
        # R: the conversion table of the new system with respect to the original one 3x3
        # c0: the origin of the new system with respect to the original one 3x1
        transform_obj = Transform()
        transform_obj.mat[:3, :3] = R
        transform_obj.translate(c0) # I use c0 directly because c0 represents the origin of the new system.
                                    # The c0 is the vector t = -R.T*d on the equation (6.13), i have already calculated t in the lookat function
        transformed = transform_obj.transform_pts(pts.T) # transform_pts takes Nx3 array as argument
        return transformed.T
 
def lookat(eye: np.ndarray, up: np.ndarray, target: np.ndarray)-> Tuple[np.ndarray, np.ndarray]:
        # Calculate the camera's view matrix (i.e., its coordinate frame transformation specified by a rotation matrix R, and a translation vector t).
        # :return a tuple containing the rotation matrix R (3 x 3) and a translation vector t (1 x 3)
        # eye: the camera's center 3x1 - C vector
        # up: the camera's up vector 3x1 - u vector
        # target: the camera's target 3x1 - K vector
        zc = (target - eye) / np.linalg.norm(target - eye) # Equation (6.6) on course's notes
        yc = (up - np.dot(up.T, zc) * zc) / np.linalg.norm(up - np.dot(up.T, zc) * zc) # Equation (6.7) on course's notes
        xc = np.cross(yc.T, zc.T) # Make the vectors row vectors equation (6.8) on course's notes
        xc = xc.flatten()
        yc = yc.flatten()
        zc = zc.flatten()
        R = np.vstack((xc, yc, zc)) # this is the R.T on (6.13)
        t = -np.dot(R, eye) # (6.13) on course's notes the -R.T * d
        t = t.T # The output is a 1x3 vector
        return R, t

def perspective_project(pts: np.ndarray, focal: float, R: np.ndarray, t: np.ndarray)-> Tuple[np.ndarray, np.ndarray]:
        # Project the specified 3d points pts on the image plane, according to a pinhole perspective projection model.
        # pts: the points to project 3xN
        # focal: the focal length of the camera
        # R: the rotation matrix to the system of the camera 3x3
        # t: the translation vector to the system of the camera 1x3
        # :return a tuple containing the projected points 2xN and the depth values 1xN
        # Calculate the camera coordinates
        
        camera_coords = world2view(pts, R, t.T) # Convert the points to the camera coordinate frame
        camera_coords = camera_coords.T # The output of world2view is a Nx3 array but i want them at 3xN
        depth_values = camera_coords[2] # The depth values are the z-coordinates of the camera coordinates
        # Apply perspective projection
        projected_pts = focal * camera_coords[:2] / depth_values # Equation (6.2) on course's notes
        
        return projected_pts, depth_values
    
def rasterize(pts_2d: np.ndarray, plane_w: int, plane_h: int, res_w: int, res_h: int)-> np.ndarray:
        # Rasterize the incoming 2d points from the camera plane to image pixel coordinates
        # pts_2d: the 2d points to rasterize 2xN
        # plane_w: the width of the camera plane
        # plane_h: the height of the camera plane
        # res_w: the width of the image plane
        # res_h: the height of the image plane
        # :return the pixel coordinates 2xN

        # Calculate the scaling factors for x and y coordinates
        scale_w = res_w / plane_w
        scale_h = res_h / plane_h
        pixel_coords = np.zeros(pts_2d.shape) 
        # Centering and Scaling the points
        # for i in range(pts_2d.shape[1]):
        #     pixel_coords[0,i] = np.around((pts_2d[0,i] + plane_w / 2) * scale_w + 0.5) 
        #     pixel_coords[1,i] = np.around((-pts_2d[1,i] + plane_h / 2) * scale_h + 0.5)

        centered_pts = pts_2d + np.array([[-plane_w / 2], [-plane_h / 2]]) # Centering the points in order to align with the bottom left corner of the image 
        scaled_coords = centered_pts * np.array([[scale_w], [scale_h]]) # Scaling the points to fit the image resolution
        pixel_coords = np.around(scaled_coords) # Rounding in order to get the nearest pixel
        return pixel_coords

def render_object(v_pos, v_clr, t_pos_idx, plane_h, plane_w, res_h, res_w, focal, eye, up, target)-> np.ndarray:
        # render the specified object from the specified camera.
        # v_pos: the coordinates of each vertex of the object Nx3
        # v_clr: the colors of a vertice of the object Nx3
        # t_pos_idx: the indices of the three vertices of triangle Fx3
        # plane_h: the height of the camera plane
        # plane_w: the width of the camera plane
        # res_h: the height of the image plane
        # res_w: the width of the image plane
        # focal: the focal length of the camera
        # eye: the camera's center 3x1
        # up: the camera's up vector 3x1
        # target: the camera's target 3x1
        # :return the rendered image res_h x res_w x 3

        # Calculate the rotation matrix R and translation vector t for the camera
        R, t = lookat(eye, up, target)
   
        # Perform perspective projection on all vertices (inside the function we go from WCS to CCS)
        projected_pts, depth_values = perspective_project(v_pos.T, focal, R, t) # The perspective_project function takes 3xN v_pos array as input
        
        # Rasterize the projected points to image pixel coordinates
        pixel_coords = rasterize(projected_pts, plane_w, plane_h, res_w, res_h)

        # Prepare data for render_img function
        faces = t_pos_idx # The indices of the three vertices of triangle Fx3
        vertices = pixel_coords.T # The pixel coordinates of each vertex of the object Nx3
        vcolors = v_clr # The colors of the vertices of the object Nx3 in RGB
        depth = depth_values.T # The depth values of the vertices of the object Nx1
        image = f.render_img(faces, vertices, vcolors, depth, "g") # The render_img function from hw1

        # Return the rendered image
        return image