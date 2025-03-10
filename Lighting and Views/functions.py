import numpy as np
# Grid
M=512
N=512
res_h = 512
res_w = 512
# VECTOR INTERPOLATION
def vector_interp(p1, p2, V1, V2, coord,dim):
    # p1, p2 are the coordinates of two points between which we want to interpolate
    # V1, V2 are the vectors at p1 and p2
    # coord is the coordinate of the point we want to interpolate 
    # dim is the dimension of the space
    if np.allclose(p1, p2):
        return np.mean([V1, V2], axis=0)
    if dim==1:
         alpha = (coord - p1[0]) / (p2[0] - p1[0])
    elif dim==2:
        alpha = (coord - p1[1] ) / (p2[1] - p1[1]) 
    # or i could use: alpha = (coord-p1[dim-1])/(p2[dim-1]-p1[dim-1])
    V = V1 * (1-alpha) + V2 * alpha
    return V

# Function which calculates some useful variables for the definition triangle 
def f_triangle(vertices):
    # vertices are the coordinates of the vertices of the triangle - 3x2 array

    # Initialization of min max arrays - 3x2 array contains: [xmin, xmax] [ymin, ymax] for each edge
    xmin_xmax_array = np.zeros((3, 2))
    ymin_ymax_array = np.zeros((3, 2))
    # Initialization of slope and b arrays
    slope = np.zeros(len(vertices))
    b = np.zeros(len(vertices))

    # Finding min and max values for each pair of vertices -> pair of vertices = edge
    for i in range(len(vertices)): # e0, e1, e2 : each edge
        pointA = vertices[i] # first vertex of the edge
        pointB = vertices[(i+1)%3] # second vertex of the edge - (i+1)%3 to connect the last vertex (vertices[2]) with the first (vertices[0])

        xmin_xmax_array[i] = [int(np.min([pointA[0], pointB[0]])), int(np.max([pointA[0], pointB[0]]))] # [xmin, xmax]
        ymin_ymax_array[i] = [int(np.min([pointA[1], pointB[1]])), int(np.max([pointA[1], pointB[1]]))] # [ymin, ymax]
       
        if pointA[0] == pointB[0]:# CASE VERTICAL LINE
            slope[i] = np.nan # the slope of the edge is infinite
            b[i] = pointA[0] # the x value of the vertical line x=b
        else: # General case
            #LINEAR EQUATION ei = slope[i] * x + b[i] <- useful for finding the active edges 
            slope[i] = (pointB[1] - pointA[1]) / (pointB[0] - pointA[0]) # the slope of each edge (yB - yA) / (xB - xA)
            b[i] = - slope[i] * pointA[0] + pointA[1]# the b parameter of the linear equation of each edge

    # Finding the min and max values of the whole triangle - the bounding box
    ymin = int(np.min(ymin_ymax_array))
    ymax = int(np.max(ymin_ymax_array))

    return xmin_xmax_array, ymin_ymax_array, slope, b, ymin, ymax

# FLAT SHADING
def f_shading(img, vertices, vcolors):
    # img is the image to be shaded with likely preexisting triangles - MxNx3 
    # vertices are the coordinates of the vertices of a triangle - 3x2 array
    # vcolors are the colors at these vertices at rgb values - 3x3 array
    
    # Calculate average color
    avg_color = np.mean(vcolors, axis=0)
    # Calculate the variables needed for the triangle definition
    xmin_xmax_array, ymin_ymax_array, slope, b, ymin, ymax = f_triangle(vertices)

    # Start the shading part using the scan lines
    for y in range(ymin, ymax +1, 1): # ymax + 1 to include the last scan line
        #INITIALIZATION
        active_edges = [] # clear the previous active edges from the previous scan line 
        active_limit_points = np.zeros((2,2)) # clear the previous active limit points
        # finding new active edges
        for i in range(len(vertices)):
            if ymin_ymax_array[i][0] <= y and ymin_ymax_array[i][1] > y: # if the scan line is between the two y values of the edge *i use > y because i want the active edges to be only 2*
                active_edges.append(i)                                   # if not use the > there are a lot cases where there are 3 active edges
        
        # Since i used > in the previous for loop i have to check for the last scan line
        if y == ymax: 
            for i in range(len(vertices)):
                if ymin_ymax_array[i][1] == y:
                    active_edges.append(i)
                # if the last scan line is horizontal
                if ymin_ymax_array[i][0] == y and ymin_ymax_array[i][1] == y:  # because i use two active edges at every case, i exclude the horizontal line from the active edges
                    active_edges.remove(i)                                     # i consider that the active edges at a triangle with a horizontal line are the other two edges

        # Finding the active limit points
        for i in range(len(active_edges)): #active_edges[i] e.g. active_edges[0] = 1 ->e1 and active_edges[1] = 2 -> e2    
            if np.isnan(slope[active_edges[i]]): # if the edge is vertical
                active_limit_points[i] = [b[active_edges[i]], y] # x = b 
            else:    
                active_limit_points[i] = [(y - b[active_edges[i]]) / slope[active_edges[i]] , y]# x = (y - b) / slope
        
        # Case: the triangle is a horizontal line (y[0] = y[1] = y[2])
        if ymin == ymax: # at the y == ymax: condition all the three edges have been added and removed from the active edges list, so the upper for loop will not define the active limit points
            active_limit_points[0] = [int(np.min(xmin_xmax_array)), y] # the first active limit point is the leftmost point of the triangle
            active_limit_points[1] = [int(np.max(xmin_xmax_array)), y] # the second active limit point is the rightmost point of the triangle

        sorted_active_limit_points = sorted(active_limit_points, key=lambda x: x[0]) # sort the active limit points by x in ascending order

        for x in range(np.ceil(np.round(sorted_active_limit_points[0][0], decimals=13)).astype(int), np.floor(np.round(sorted_active_limit_points[1][0], decimals=13)).astype(int) + 1, 1): # for every pixel between the two active limit points
            # since i want to include only the pixels that they are inside the triangle i use the np.ceil and np.floor functions
            # also i use the np.round function to avoid the floating point errors
            img[y, x] = avg_color # shade the pixel with the average color of the triangle
    return img # return the updated image


# GOURAUD SHADING
def gouraud_shading(img, vertices, vcolors):
    # img is the image to be shaded with likely preexisting triangles - MxNx3 
    # vertices are the coordinates of the vertices of a triangle - 3x2 array
    # vcolors are the colors at these vertices at rgb values - 3x3 array

    # Calculate the variables needed for the triangle definition
    xmin_xmax_array, ymin_ymax_array, slope, b, ymin, ymax = f_triangle(vertices)

    # Case: the triangle is a point
    if ymin == ymax and int(np.min(xmin_xmax_array)) == int(np.max(xmin_xmax_array)):
        img[int(vertices[0][1]), int(vertices[0][0])] = np.mean(vcolors, axis=0) # if the triangle is a point then the shading is the same as the flat shading
    else:
        # Start the shading part using the scan lines
        for y in range(ymin, ymax + 1, 1):
            # Case: the triangle is a horizontal line (y[0] = y[1] = y[2])
            if ymax == ymin:
                sorted_indices_asc = np.argsort(vertices[:, 0]) # sort the vertices by x in ascending order and "return" the indices of the sorted array 
                sorted_vertices = vertices[sorted_indices_asc] # sorted vertices
                sorted_vcolors = vcolors[sorted_indices_asc] # sorted vcolors based on the sorted vertices 
                
                if sorted_vertices[0][0] == sorted_vertices[1][0] and sorted_vcolors[0][0] != sorted_vcolors[1][0]:
                    concatenated_array = np.concatenate([sorted_vcolors[0][np.newaxis, :], sorted_vcolors[1][np.newaxis, :]], axis=0) 
                    for x in range (int(sorted_vertices[0][0]), int(sorted_vertices[2][0]) + 1, 1):
                        img[y, x] = vector_interp(sorted_vertices[0], sorted_vertices[2], np.mean(concatenated_array, axis=0), sorted_vcolors[2], x, 1)
                
                elif sorted_vertices[1][0] == sorted_vertices[2][0] and sorted_vcolors[1][0] != sorted_vcolors[2][0]:
                    concatenated_array = np.concatenate([sorted_vcolors[1][np.newaxis, :], sorted_vcolors[2][np.newaxis, :]], axis=0)
                    for x in range (int(sorted_vertices[0][0]), int(sorted_vertices[2][0]) + 1, 1):
                        img[y, x] = vector_interp(sorted_vertices[0], sorted_vertices[2], sorted_vcolors[0], np.mean(concatenated_array, axis=0), x, 1)
                
                else:
                    # shade the pixels between the first two vertices
                    for x in range (int(sorted_vertices[0][0]), int(sorted_vertices[1][0]) + 1, 1):
                        img[y, x] = vector_interp(sorted_vertices[0], sorted_vertices[1], sorted_vcolors[0], sorted_vcolors[1], x, 1)
                    # shade the pixels between the second and the third vertex
                    for x in range (int(sorted_vertices[1][0]), int(sorted_vertices[2][0]) + 1, 1):
                        img[y, x] = vector_interp(sorted_vertices[1], sorted_vertices[2], sorted_vcolors[1], sorted_vcolors[2], x, 1)
            else:
                # General Case
                #INITIALIZATION
                active_edges = [] # clear the previous active edges from the previous scan line
                active_limit_points = np.zeros((2,2)) # clear the previous active limit points
                color_alp = np.zeros((2, 3))
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
                    color_alp[i] = vector_interp(vertices[active_edges[i]], vertices[(active_edges[i]+1)%3], vcolors[active_edges[i]], vcolors[(active_edges[i]+1)%3], y, 2)

                sorted_active_limit_points = sorted(active_limit_points, key=lambda x: x[0]) # sort the active limit points by x in ascending order
        
                for x in range(np.ceil(np.round(sorted_active_limit_points[0][0], decimals=13)).astype(int), np.floor(np.round(sorted_active_limit_points[1][0], decimals=13)).astype(int) + 1, 1): # for every pixel in the scan line
                    img[y, x] = vector_interp(active_limit_points[0], active_limit_points[1], color_alp[0], color_alp[1], x, 1) # 1D because the scan line is horizontal
    
    return np.clip(img, 0, 1) # return the updated image


# RENDER IMAGE
def render_img(faces,vertices,vcolors,depth,shading):
    # img is the image which is being shaded - MxNx3 array - contains K colored triangles which forms the 3D projection of an object
    # faces is an array w the three vertices which construct a triangle
    #   faces = [[0, 1, 2], [3, 2, 4], ...] hich contains the id of the vertices of the K triangles - Kx3 array - each row contains the ids of
    #   i = 0: triangle: C1 = vertices[0], C2 = vertices[1], C3 = vertices[2]
    #   i = 1: triangle: C1 = vertices[3], C2 = vertices[2], C3 = vertices[4]
    # vertices is an array of the vertices of the object              - Lx2 array - each row contains the coordinates of the vertex - # * all of the vertices are inside the canvas
    # vcolors is an array of the colors of the vertices of the object - Lx3 array - each row contains the RGB values of the corresponding vertex
    # depth is an array of the depth of the each vertex               - Lx1 array
    # shading is the shading method to be used                        - string - "f" for flat shading, "g" for gouraud shading

    #img = np.ones((M,N,3)) # create a white image with MxNx3 dimensions
    img = np.ones((res_h, res_w, 3)) # create a whit image with MxNx3 dimensions
    K = len(faces)

    # Finding the depth of each triangle
    triangle_depth = np.zeros(K)
    for i in range(K):
        triangle_depth[i] = np.mean(depth[faces[i]])

    # Sorting the array by ascending depth and the faces array accordingly
    sorted_indices_asc = np.argsort(triangle_depth) # the indices of the sorted array - i use them to sort both the triangle_depth and the faces array
    sorted_indices_desc = sorted_indices_asc[::-1]  # the indices of the sorted array in descending order
    sorted_triangle_depth = triangle_depth[sorted_indices_desc] # the sorted triangle depth array
    sorted_faces = faces[sorted_indices_desc] # the sorted faces array

    vertices_triangle = np.empty((K, 3, 2))  # K triangles, each with 3 vertices, each with 2 coordinates
    vcolors_triangle = np.empty((K, 3, 3))   # K triangles, each with 3 vertices, each with 3 color components

    for i in range(K):
        # Defining the vertices and the colors of the current triangle
            vertices_triangle[i] = vertices[sorted_faces[i]]
            vcolors_triangle[i] = vcolors[sorted_faces[i]]
        # Shading the current triangle
            if shading == "f":
                img = f_shading(img, vertices_triangle[i], vcolors_triangle[i])
            elif shading == "g":
                img = gouraud_shading(img, vertices_triangle[i], vcolors_triangle[i])
    return img