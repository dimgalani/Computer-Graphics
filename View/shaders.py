import numpy as np
import functions as f
import lighting as l
from functions import *

def shade_gouraud(verts_p, verts_n, verts_c, b_coords, cam_pos, ka, kd, ks, n, l_pos, l_int, l_amb, X):
    # verts_p: 2 x 3, the 2D coordinates of the vertices of the triangle
    # verts_n: 3 x 3, the normal vectors of the vertices of the triangle - each column is a normal vector
    # verts_c: 3 x 3, the RGB color of the vertices of the triangle - each column is a color
    # b_coords: 3 x 1, the barycentric coordinates of the triangle
    # cam_pos: 3 x 1, the coordinates of the camera/viewer
    # ka: number, the ambient reflection coefficient
    # kd: number, the diffuse reflection coefficient
    # ks: number, the specular reflection coefficient
    # n: number, the Phong constant
    # l_pos: N x 3, the positions of the light sources
    # l_int: N x 3, the intensities of the light sources
    # l_amb: 1 x 3, the ambient light intensity of the scene
    # X: M x N x 3, the image with preexisting colored triangles
    # returns: Y: M x N x 3, the image with the triangle shaded using Gouraud shading

    # Calculate the ambient light component
    ambient_component = ka * l_amb

    # For each vertex of the triangle, find the lighting
    for i in range(verts_c.shape[0]):
        verts_c[:, i] = l.light(b_coords, verts_n[:, i].T, verts_c[:, i].T, cam_pos, ambient_component, kd, ks, n, l_pos, l_int).T

    vertices = verts_p.T
    vcolors = verts_c.T

    # Call gouraud_shading 
    Y = f.gouraud_shading(X, vertices, vcolors)
    return Y

def shade_phong(verts_p, verts_n, verts_c, b_coords, cam_pos, ka, kd, ks, n, l_pos, l_int, l_amb, X):
    # verts_p: 2 x 3, the 2D coordinates of the vertices of the triangle
    # verts_n: 3 x 3, the normal vectors of the vertices of the triangle - each column is a normal vector
    # verts_c: 3 x 3, the RGB color of the vertices of the triangle - each column is a color
    # b_coords: 3 x 1, the barycentric coordinates of the triangle
    # cam_pos: 3 x 1, the coordinates of the camera/viewer
    # ka: number, the ambient reflection coefficient
    # kd: number, the diffuse reflection coefficient
    # ks: number, the specular reflection coefficient
    # n: number, the Phong constant
    # l_pos: N x 3, the positions of the light sources
    # l_int: N x 3, the intensities of the light sources
    # l_amb: 1 x 3, the ambient light intensity of the scene
    # X: M x N x 3, the image with preexisting colored triangles
    # returns: Y: M x N x 3, the image with the triangle shaded using Phong shading

    ambient_component = ka * l_amb

    vertices = verts_p.T
    vcolors = verts_c.T
    normalvecs = verts_n.T
    img = X

    # img is the image to be shaded with likely preexisting triangles - MxNx3 
    # vertices are the coordinates of the vertices of a triangle - 3x2 array
    # vcolors are the colors at these vertices at rgb values - 3x3 array

    # Calculate the variables needed for the triangle definition
    xmin_xmax_array, ymin_ymax_array, slope, b, ymin, ymax = f_triangle(vertices)

    # Case: the triangle is a point
    if ymin == ymax and int(np.min(xmin_xmax_array)) == int(np.max(xmin_xmax_array)):
        # Compute the light from one vertex
        mean_vcolor = np.mean(vcolors, axis=0)
        color_light = l.light(b_coords, verts_n[:, 0].T, mean_vcolor, cam_pos, ambient_component, kd, ks, n, l_pos, l_int)
        img[int(vertices[0][1]), int(vertices[0][0])] = color_light # if the triangle is a point then the shading is the same as the flat shading
    else:

        # Start the shading part using the scan lines
        for y in range(ymin, ymax + 1, 1):
            # Case: the triangle is a horizontal line (y[0] = y[1] = y[2])
            if ymax == ymin:
                sorted_indices_asc = np.argsort(vertices[:, 0]) # sort the vertices by x in ascending order and "return" the indices of the sorted array 
                sorted_vertices = vertices[sorted_indices_asc] # sorted vertices
                sorted_vcolors = vcolors[sorted_indices_asc] # sorted vcolors based on the sorted vertices
                sorted_normals = normalvecs[sorted_indices_asc] 
                
                # Case the first two vertices have the same x value but different colors
                if sorted_vertices[0][0] == sorted_vertices[1][0] and sorted_vcolors[0][0] != sorted_vcolors[1][0]:
                    concatenated_array = np.concatenate([sorted_vcolors[0][np.newaxis, :], sorted_vcolors[1][np.newaxis, :]], axis=0)
                    # concatenated_normals = np.concatenate([sorted_normals[0][np.newaxis, :], sorted_normals[1][np.newaxis, :]], axis=0)
                    for x in range (int(sorted_vertices[0][0]), int(sorted_vertices[2][0]) + 1, 1):
                        interp_color = vector_interp(sorted_vertices[0], sorted_vertices[2], sorted_vcolors[0], np.mean(concatenated_array, axis=0), x, 1)
                        interp_normal = vector_interp(sorted_vertices[0], sorted_vertices[2], sorted_normals[0], sorted_normals[2], x, 1)
                        light = l.light(b_coords, interp_normal, interp_color, cam_pos, ambient_component, kd, ks, n, l_pos, l_int) 
                        img[y, x] = light
                
                elif sorted_vertices[1][0] == sorted_vertices[2][0] and sorted_vcolors[1][0] != sorted_vcolors[2][0]:
                    concatenated_array = np.concatenate([sorted_vcolors[1][np.newaxis, :], sorted_vcolors[2][np.newaxis, :]], axis=0)
                    for x in range(int(sorted_vertices[0][0]), int(sorted_vertices[2][0]) + 1, 1):
                        interp_color = vector_interp(sorted_vertices[0], sorted_vertices[2], sorted_vcolors[0], np.mean(concatenated_array, axis=0), x, 1)
                        interp_normal = vector_interp(sorted_vertices[0], sorted_vertices[2], sorted_normals[0], sorted_normals[2], x, 1)
                        light = l.light(b_coords, interp_normal, interp_color, cam_pos, ambient_component, kd, ks, n, l_pos, l_int)
                        img[y, x] = light  # Apply lighting to color
                else:
                    # shade the pixels between the first two vertices
                    for x in range(int(sorted_vertices[0][0]), int(sorted_vertices[1][0]) + 1, 1):
                        interp_color = vector_interp(sorted_vertices[0], sorted_vertices[1], sorted_vcolors[0], sorted_vcolors[1], x, 1)
                        interp_normal = vector_interp(sorted_vertices[0], sorted_vertices[1], sorted_normals[0], sorted_normals[1], x, 1)
                        light = l.light(b_coords, interp_normal, interp_color, cam_pos, ambient_component, kd, ks, n, l_pos, l_int)
                        img[y, x] =  light  # Apply lighting to color
                    # shade the pixels between the second and the third vertex
                    for x in range(int(sorted_vertices[1][0]), int(sorted_vertices[2][0]) + 1, 1):
                        interp_color = vector_interp(sorted_vertices[1], sorted_vertices[2], sorted_vcolors[1], sorted_vcolors[2], x, 1)
                        interp_normal = vector_interp(sorted_vertices[1], sorted_vertices[2], sorted_normals[1], sorted_normals[2], x, 1)  # Interpolate normal
                        light = l.light(b_coords, interp_normal, interp_color, cam_pos, ambient_component, kd, ks, n, l_pos, l_int)  # Compute lighting
                        img[y, x] = light  # Apply lighting to color
            else:
                # General Case
                #INITIALIZATION
                active_edges = [] # clear the previous active edges from the previous scan line
                active_limit_points = np.zeros((2,2)) # clear the previous active limit points
                color_alp = np.zeros((2, 3))
                normal = np.zeros((2, 3))
                # finding new active edges
                for i in range(len(vertices)):
                    if ymin_ymax_array[i][0] <= y and ymin_ymax_array[i][1] > y: # if the scan line is between the two y values of the edge
                        active_edges.append(i)   # add the id of the edge to the active edges

                # Since i used > in the previous for loop i have to check for the last scan line
                # Same as flat_shading method
                if y == ymax:
                   for i in range(len(vertices)):
                        if ymin_ymax_array[i][1] == y:
                            active_edges.append(i)
                        # if the last scan line is horizontal    
                        if ymin_ymax_array[i][0] == y and ymin_ymax_array[i][1] == y:
                            active_edges.remove(i)
        
                # Finding the active limit points
                for i in range(len(active_edges)): #active_edges[i] e.g. active_edges[0] = 1 ->e1 and active_edges[1] = 2 -> e2    
                    if np.isnan(slope[active_edges[i]]): # if the edge is vertical
                        active_limit_points[i] = [b[active_edges[i]], y] # x = b 
                    else:    
                        active_limit_points[i] = [(y - b[active_edges[i]]) / slope[active_edges[i]] , y]# x = (y - b) / slope

                for i in range(2):
                    # Interpolate the color and the normal vector of the active limit points
                    color_alp[i] = vector_interp(vertices[active_edges[i]], vertices[(active_edges[i]+1)%3], vcolors[active_edges[i]], vcolors[(active_edges[i]+1)%3], y, 2)
                    normal[i] = vector_interp(vertices[active_edges[i]], vertices[(active_edges[i]+1)%3], normalvecs[active_edges[i]], normalvecs[(active_edges[i]+1)%3], y, 2)

                sorted_active_limit_points = sorted(active_limit_points, key=lambda x: x[0]) # sort the active limit points by x in ascending order
        
                for x in range(np.ceil(np.round(sorted_active_limit_points[0][0], decimals=13)).astype(int), np.floor(np.round(sorted_active_limit_points[1][0], decimals=13)).astype(int) + 1, 1): # for every pixel in the scan line
                    interp_color = vector_interp(active_limit_points[0], active_limit_points[1], color_alp[0], color_alp[1], x, 1)
                    interp_normal = vector_interp(active_limit_points[0], active_limit_points[1], normal[0], normal[1], x, 1)
                    # Compute the light from the interpolated point
                    color_light = l.light(b_coords, interp_normal, interp_color, cam_pos, ambient_component, kd, ks, n, l_pos, l_int)
                    img[y, x] = np.clip(color_light, 0, 1)
    return np.clip(img, 0, 1) # return the updated image