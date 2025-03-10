import numpy as np
from functions import vector_interp, gouraud_shading
import transformations as trans
from shaders import *
import lighting as l

def bilerp(uv, texture_map):
    # uv: 1x2, the uv coordinates of the point
    # texture_map: MxMx3, the texture map
    # returns: 1x3, the color of the point
    M, N, _ = texture_map.shape
    
    # Calculate the pixel coordinates
    x = uv[0] * (N - 1)
    y = uv[1] * (M - 1)

    # Get the four surrounding pixel coordinates
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, N - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, M - 1)

    # Get the values at the corners of the rectangle of the area that the point belongs to in the texture map
    f00 = texture_map[y0, x0] # the pixel
    f01 = texture_map[y1, x0] # the upper pixel
    f10 = texture_map[y0, x1] # the right pixel
    f11 = texture_map[y1, x1] # the upper right pixel
    
    # Interpolate along x-axis at the bottom (y0)
    Vx0 = vector_interp([x0, y0], [x1, y0], f00, f10, x, 1)
    
    # Interpolate along x-axis at the top (y1)
    Vx1 = vector_interp([x0, y1], [x1, y1], f01, f11, x, 1)
    
    # Interpolate along y-axis using the results from x-axis interpolation
    color = vector_interp([x0, y0], [x0, y1], Vx0, Vx1, y, 2)
    return color

def render_object_map(shader, focal, eye, lookat, up, bg_color, M, N, H, W,
                  verts, vert_colors, faces, ka, kd, ks, n, lpos, lint, light_amb, uvs, uvs_faces, texture_map):
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
    #! NEW ARGUMENTS
    # uvs: 2 x Nverts, the uv coordinates of the vertices of the object
    # uvs_faces: 3 x Nfaces, the uv coordinates of the triangles of the object
    # texture_map: A x B x 3, the texture map

    #* 1) Calculate the normals of the vertices
    normals = l.calculate_normals(verts, faces)
    #* 2) Project the vertices onto the camera plane using the perspective_project function from transformations.py
    # The triangles outside the camera plane are not visible thus are not colored
    # Calculate the rotation matrix and translation vector
    R, t = trans.lookat(eye, up, lookat)
    # Project the vertices
    projected_verts, depth_values = trans.perspective_project(verts, focal, R, t) 
    # projected_verts: 2 x Nv and depth_values: 1 x Nv
    # Rasterize the projected vertices to image pixel coordinates
    pixel_coords = trans.rasterize(projected_verts, W, H, N, M).astype(int)
    # pixel_coords: 2 x Nv
    # Initialize the image
    img = np.full((M, N, 3), bg_color) # Create an image with MxNx3 dimensions with the backround color
    # Compute triangle_depth
    triangle_depth = np.mean(depth_values[faces], axis=0)
    # Sorting the array by descending depth
    sorted_triangles_asc = np.argsort(triangle_depth)
    sorted_triangles_desc = sorted_triangles_asc[::-1]

    for triangle in sorted_triangles_desc:

        vertices_triangle = faces[:, triangle] # size 3x1 - the indices of the vertices of the triangle
        vertices_triangle2d = pixel_coords[:, vertices_triangle].T # 3x2 - the pixel coordinates of the vertices of the triangle
        # pixel_coords 2xNv so vertices_triangle2d are 3x2
        bcoords = np.mean(verts[:, vertices_triangle], axis=0).T

        uv_triangle = uvs_faces[triangle, :] # trigwno sto texture map? to trigwno antistoizetai akribws??
        uv_coords = uvs[:, uv_triangle].T # 3x2 - the uv coordinates of the vertices of the triangle

        vcolors_triangle = np.array([
            bilerp(uv_coords[0], texture_map),
            bilerp(uv_coords[1], texture_map),
            bilerp(uv_coords[2], texture_map)
        ]).T
        
        if shader == "gouraud":
            img = shade_gouraud(vertices_triangle2d.T, normals[:, vertices_triangle], vcolors_triangle, bcoords, eye, ka, kd, ks, n, lpos, lint, light_amb, img)
        elif shader == "phong":
            img = shade_phong(vertices_triangle2d.T, normals[:, vertices_triangle], vcolors_triangle, bcoords, eye, ka, kd, ks, n, lpos, lint, light_amb, img)
        elif shader == "nolight":
            img = gouraud_shading(img, vertices_triangle2d, vcolors_triangle.T)
        else:
            raise ValueError("Invalid shader type. Choose 'gouraud' or 'phong' or 'nolight'.")
    return img