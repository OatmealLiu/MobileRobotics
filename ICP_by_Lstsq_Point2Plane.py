import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from math import sin, cos, atan2, pi
from functools import partial


"""
Generate data
"""
angle = pi / 4
R_true = np.array([[cos(angle), -sin(angle)], 
                   [sin(angle),  cos(angle)]])
t_true = np.array([[-2], [5]])

# Generate data as a list of 2d points
num_points = 30
true_data = np.zeros((2, num_points))
true_data[0, :] = range(0, num_points)
true_data[1, :] = 0.2 * true_data[0, :] * np.sin(0.5 * true_data[0, :]) 

# Move the data
moved_data = R_true.dot(true_data) + t_true

Q = true_data
P = moved_data

"""
Find data association
"""
def get_correspondence_indices(P, Q):
    """For each point in P find closest one in Q."""
    p_size = P.shape[1]
    q_size = Q.shape[1]
    correspondences = []
    for i in range(p_size):
        p_point = P[:, i]
        min_dist = sys.maxsize
        chosen_idx = -1
        for j in range(q_size):
            q_point = Q[:, j]
            dist = np.linalg.norm(q_point - p_point)
            if dist < min_dist:
                min_dist = dist
                chosen_idx = j
        correspondences.append((i, chosen_idx))
    return correspondences

def draw_correspondences(P, Q, correspondences, ax):
    label_added = False
    for i, j in correspondences:
        x = [P[0, i], Q[0, j]]
        y = [P[1, i], Q[1, j]]
        if not label_added:
            ax.plot(x, y, color='grey', label='correpondences')
            label_added = True
        else:
            ax.plot(x, y, color='grey')
    ax.legend()

"""
Centering data to the center of mass
"""
def center_data(data, exclude_indices=[]):
    reduced_data = np.delete(data, exclude_indices, axis=1)
    center = np.array([reduced_data.mean(axis=1)]).T
    return center, data - center

"""
Define 2D rotation matrix
"""
def dR(theta):
    return np.array([[-sin(theta), -cos(theta)],
                     [cos(theta),  -sin(theta)]])
def R(theta):
    return np.array([[cos(theta), -sin(theta)],
                     [sin(theta), cos(theta)]])

"""
Calc Jacobian
"""
def jacobian(x, p_point):
    theta = x[2]
    J = np.zeros((2, 3))
    J[0:2, 0:2] = np.identity(2)
    J[0:2, [2]] = dR(theta).dot(p_point)
    return J

"""
Calc single error
"""
def error(x, p_point, q_point):
    rotation = R(x[2])
    translation = x[0:2]
    prediction = rotation.dot(p_point) + translation
    return prediction - q_point


"""
Calc normals for target plane
"""
def compute_normals(points, step=1):
    normals = [np.array([[0, 0]])]
    normals_at_points = []
    for i in range(step, points.shape[1] - step):
        prev_point = points[:, i - step]
        next_point = points[:, i + step]
        curr_point = points[:, i]
        dx = next_point[0] - prev_point[0] 
        dy = next_point[1] - prev_point[1]
        normal = np.array([[0, 0],[-dy, dx]])
        normal = normal / np.linalg.norm(normal)
        normals.append(normal[[1], :])  
        normals_at_points.append(normal + curr_point)
    normals.append(np.array([[0, 0]]))
    return normals, normals_at_points


def plot_normals(normals, ax):
    label_added = False
    for normal in normals:
        if not label_added:
            ax.plot(normal[:,0], normal[:,1], color='grey', label='normals')
            label_added = True
        else:
            ax.plot(normal[:,0], normal[:,1], color='grey')
    ax.legend()
    return ax


"""
Calc:
    - single error          e with normal
    - Jacobian              J with normal
    - Hessian Matrix        H
    - Gradient Matrix       g
    - loss function value   chi
"""
def prepare_system_normals(x, P, Q, correspondences, normals):
    H = np.zeros((3,3))
    g = np.zeros((3,1))
    chi = 0
    for (i, j), normal in zip(correspondences, normals):
        p_point = P[:, [i]]
        q_point = Q[:, [j]]
        
        # 1. calc error
        e = normal.dot(error(x, p_point, q_point)) # !!!
        # 2. calc Jacobian
        J = normal.dot(jacobian(x, p_point))       # !!!
        # 3. calc Hessian and update
        H += J.T.dot(J)
        # 4. calc gradient and update
        g += J.T.dot(e)
        # 5. add error on loss function value
        chi += e.T * e
    return H, g, chi

"""
kernel: reject outliers
"""
def kernel(threshold, error):
    if np.linalg.norm(error) < threshold:
        return 1.0
    return 0.0

"""
ICP based on least squares method with Point to Plane association

"""

def icp_normal(P, Q, normals, iterations=20):
    # Initialize pose x = {x,y,theta}
    x = np.zeros((3,1))
    chi_values = []
    x_values = [x.copy()]
    P_values = [P.copy()]
    P_latest = P.copy()
    corresp_values = []
    
    for i in range(iterations):
        # 1. find correspondences
        correspondences = get_correspondence_indices(P_latest, Q)
        corresp_values.append(correspondences)
        
        # 2. calc Hessian Matrix, Gradient Matrix and loss function value
        H, g, chi = prepare_system_normals(x, P, Q, correspondences, normals)
        
        # 3. run least-square solver to find incremental dx
        dx = np.linalg.lstsq(H, -g, rcond=None)[0]
        
        # 4. update pose x
        x += dx
        x[2] = atan2(sin(x[2]), cos(x[2])) # normalize angle
        
        # 6. record current loss function value
        chi_values.append(chi.item(0))     # add error to list of errors
        x_values.append(x.copy())
        
        # 7. do Rot-Translate on P set
        rot = R(x[2])
        t = x[:2]
        P_latest = rot.dot(P.copy()) + t
        P_values.append(P_latest)
        
    corresp_values.append(corresp_values[-1])
    return P_values, chi_values, corresp_values