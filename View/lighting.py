import numpy as np

# The lighting of one point which is calculated by the Phong model
def light(point, normal, vcolor, cam_pos, ka, kd, ks, n, l_pos, l_int):
    # point: 1 x 3, the coordinates of the point
    # normal: 1 x 3, the normal vector at the point
    # vcolor: 1 x 3, the RGB color of the point
    # cam_pos: 1 x 3, the coordinates of the camera
    # ka: number, the ambient reflection coefficient
    # kd: number, the diffuse reflection coefficient
    # ks: number, the specular reflection coefficient
    # n: number, the Phong constant
    # l_pos: N x 3, the positions of the light sources
    # l_int: N x 3, the intensities of the light sources
    
    # Initialize the color components
    Ia = np.zeros(3)
    Id = np.zeros(3)
    Is = np.zeros(3)

    # Calculate the ambient component (constant for all light sources)   
    Ia = ka
    l_pos = np.atleast_2d(l_pos)
    
    # Iterate over each light source
    for i in range(len(l_pos)):
        # L is the light direction vector
        L = l_pos[i] - point
        L = L / np.linalg.norm(L)

        # V is the view direction vector
        V = cam_pos - point  
        V = V / np.linalg.norm(V)  

        # R is the reflection direction vector
        R = 2 * np.dot(L, normal) * normal - L
        R = R / np.linalg.norm(R)

        # Diffuse component
        diff_intensity = kd * max(np.dot(L, normal), 0)
        Id += diff_intensity * l_int[i] * vcolor

        # Specular component
        spec_intensity = ks * np.dot(V, R.T) ** n
        Is += spec_intensity * l_int[i] * vcolor
    
    # Sum all components to get the final color
    final_color = Ia + Id + Is
    # Ensure the final color is within the [0, 1] range
    final_color = np.clip(final_color, 0, 1)
    return final_color

def calculate_normals(verts, faces):
    # verts: 3 x Nv, the coordinates of the vertices of the object
    # faces: 3 x NT, the triangles of the object (each column has the ascending indices of the kth triangle), 1 ≤ k ≤ NT
    # returns: 3 x Nv, the normal vectors of the vertices
    Nv = verts.shape[1]
    NT = faces.shape[1]

    # Initialize the normals array
    normals = np.zeros((3, Nv))
    
    # Calculate normals for each face and accumulate
    for i in range(NT):
        # Get the vertex indices for the current face (adjusted for 0-indexing in Python)
        v0, v1, v2 = faces[:, i]
        
        # Get the vertex coordinates
        p0, p1, p2 = verts[:, v0], verts[:, v1], verts[:, v2]
        
        # Compute the two edge vectors
        edge1 = p1 - p0
        edge2 = p2 - p0
        
        # Compute the face normal using the cross product
        face_normal = np.cross(edge1, edge2)
        
        # Normalize the face normal to ensure it's a unit vector
        face_normal = face_normal / np.linalg.norm(face_normal)
        
        # Accumulate the face normal to the vertex normals
        normals[:, v0] += face_normal
        normals[:, v1] += face_normal
        normals[:, v2] += face_normal

    normals = normals / np.linalg.norm(normals, axis=0)

    return normals