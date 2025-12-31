import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

#Quaternion properties and operations

def normalize(q):
    return q / np.linalg.norm(q)

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def rotate_vector(q, v):
    """Rotate 3D vector v using quaternion q"""
    qv = np.array([0, *v]) #converts 3D vector to vector quaternion
    return quat_mult(quat_mult(q, qv), quat_conjugate(q))[1:] #rotaion operation


#Cube Geometry

cube = np.array([
    [-1, -1, -1],
    [ 1, -1, -1],
    [ 1,  1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
    [ 1, -1,  1],
    [ 1,  1,  1],
    [-1,  1,  1]
])

edges = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7)
]

#Animation Setup

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim([-2,2])
ax.set_ylim([-2,2])
ax.set_zlim([-2,2])
ax.set_box_aspect([1,1,1])

lines = [ax.plot([], [], [], 'k')[0] for _ in edges]

q = np.array([1,0,0,0])  # identity quaternion 

axis = normalize(np.array([1, 1, 0.3])) # rotation axis or u in q = sin(theta/2) + u*cos(theta/2)
theta = 0.3
dq = normalize(np.array([np.cos(theta/2), *(axis*np.sin(theta/2))])) #dq for incremental rotation

def update(frame):
    global q
    q = normalize(quat_mult(dq, q))#take the existing cube rotationthen rotate it a little more

    rotated = np.array([rotate_vector(q, v) for v in cube]) #finally using the quaternion operation to rotate each vertex

    for (i,(a,b)) in enumerate(edges):
        xs = [rotated[a,0], rotated[b,0]]
        ys = [rotated[a,1], rotated[b,1]]
        zs = [rotated[a,2], rotated[b,2]]
        lines[i].set_data(xs, ys)
        lines[i].set_3d_properties(zs)

    return lines

ani = FuncAnimation(fig, update, frames=500, interval=30, blit=True)
plt.title("Quaternion Rotation Demo (No Rotation Matrices)")
plt.show()