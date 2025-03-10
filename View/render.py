import numpy as np
import transformations as trans
from shaders import *
import lighting as l

def render_object(shader, focal, eye, lookat, up, bg_color, M, N, H, W,
                  verts, vert_colors, faces, ka, kd, ks, n, lpos, lint, light_amb):
    # shader: string {"gouraud", "phong"} deciding the coloring function
    # focal: number, the distance of the projection from the centre of the camera
    # eye:   3 x 1, the coordinates of the centre of the camera
    # lookat:3 x 1, the coordinates of the camera target point
    # up:    3 x 1, unit up vector of the camera
    # bg_color: 3 x 1, the RGB color of the background
    # M, N:  height and width of the image in pixels M x N pixels
    # H, W:  physical height and width of the camera
    # verts: 3 x Nv, the coordinates of the vertices of the object
    # vert_colors: 3 x Nv, the RGB color of each vertex of the object
    # faces: 3 x NT, the triangles of the object (each column has the ascending indices of the kth triangle), 1 ≤ k ≤ NT
    # ka: number, the ambient reflection coefficient
    # kd: number, the diffuse reflection coefficient
    # ks: number, the specular reflection coefficient
    # n:  number, the Phong constant
    # lpos: N x 3, the positions of the light sources
    # lint: N x 3, the intensities of the light sources
    # light_amb: 3 x 1, the ambient light intensity

    #* 1) Calculate the normals of the vertices
    normals = l.calculate_normals(verts, faces)

    #* 2) Project the vertices onto the camera plane using the perspective_project function from transformations.py
    # Calculate the rotation matrix and translation vector
    R, t = trans.lookat(eye, up, lookat)

    # Project the vertices
    projected_verts, depth_values = trans.perspective_project(verts, focal, R, t) 
    # projected_verts: 2 x Nv and depth_values: 1 x Nv

    # Rasterize the projected vertices to image pixel coordinates
    pixel_coords = trans.rasterize(projected_verts, W, H, N, M).astype(int)
    # pixel_coords: 2 x Nv

    #! new render function, i cannot use the old because i want to include the phong shading

    # Initialize the image
    img = np.full((M, N, 3), bg_color) # Create an image with MxNx3 dimensions with the backround color

    # Compute triangle_depth
    triangle_depth = np.mean(depth_values[faces], axis=0)

    # Sorting the array by descending depth
    sorted_triangles_asc = np.argsort(triangle_depth)
    sorted_triangles_desc = sorted_triangles_asc[::-1]

    for triangle in sorted_triangles_desc:

        vertices_triangle = faces[:, triangle] # size 3x1
      
        vertices_triangle2d = pixel_coords[:, vertices_triangle].T
        # pixel_coords 2xNv so vertices_triangle2d are 3x2
        vcolors_triangle = vert_colors[vertices_triangle, :].T
        bcoords = np.mean(verts[:, vertices_triangle], axis=0).T

        if shader == "gouraud":
            img = shade_gouraud(vertices_triangle2d.T, normals[:, vertices_triangle], vcolors_triangle, bcoords, eye, ka, kd, ks, n, lpos, lint, light_amb, img)
        elif shader == "phong":
            img = shade_phong(vertices_triangle2d.T, normals[:, vertices_triangle], vcolors_triangle, bcoords, eye, ka, kd, ks, n, lpos, lint, light_amb, img)
        else:
            raise ValueError("Invalid shader type. Choose 'gouraud' or 'phong'.")
    return img