import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy import ndimage
import cv2

REAL_DIST_AB = 25.5
def run_compute_table_size():
        a = np.array([2724, 2863, 1])
        b = np.array([1310, 1743, 1])
        a_tag = np.array([3906, 1792, 1])
        b_tag = np.array([2363, 1259, 1])
        c = np.array([166, 857, 1])
        line_ab = do_cross(a, b)
        line_atbt = do_cross(a_tag, b_tag)
        d = do_cross(line_atbt, line_ab)
        T = dist(c, a) * dist(d, b)/(dist(b, a) * dist(d, c)) * REAL_DIST_AB
        print(f"size of table is {np.round(T, 1)} cm, vs real size of 110 cm")
def dist(x, y):
        return np.linalg.norm(x[:-1] - y[:-1])
def do_cross(a, b):
        line_ab = np.cross(a, b).astype(float)
        line_ab /= line_ab[-1]
        return line_ab

def analize_circule_parabula(points):
        # a = [251, 2723]
        # b = [875, 1884]
        # c = [1935, 1229]
        # d = [2923, 1135]
        # e = [3717, 1321]
        # f = [1382, 1497]
        # points = np.array([a, b, c, d, e, f])
        A = np.zeros((len(points), 6))
        for i, (x, y) in enumerate(points):
                A[i] = [x ** 2, x * y, y ** 2, x, y, 1]
        U, S, V = np.linalg.svd(A)
        a, b, c, d, e, f = V[-1]
        print(f'Conic equation: {a}x^2 + {b}xy + {c}y^2\n + {d}x + {e}y + {f} = 0')
        discriminant = b ** 2 - 4 * a * c
        print(f"discriminant is {discriminant}")
        if np.abs(discriminant) <= 1e-10:
                print("this conic is a parabula")


def find_fundamental_matrix(world_points, image_points):
        n = world_points.shape[0]
        # Construct the matrix A
        A = np.zeros((2 * n, 12))
        for i in range(n):
                x, y, z = world_points[i][:-1]
                u, v = image_points[i][:-1]
                A[2 * i] = [x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u]
                A[2 * i + 1] = [0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v]
        # Solve for the nullspace of A
        _, s, V = np.linalg.svd(A)
        P = V[-1].reshape(3, 4)
        P /= P[-1, -1]
        return P


def decompose_camera_matrix(P):
        # Compute the camera center
        C = -np.linalg.inv(P[:, :3]) @ P[:, 3]
        # Compute the rotation matrix
        M = P[:, :3]
        Q, R = np.linalg.qr(np.linalg.inv(M))
        K = np.linalg.inv(R)
        R = Q.T
        # Ensure that the diagonal elements of K are positive
        if np.linalg.det(R) < 0:
                R = -R
                K = -K
        # Compute the translation vector
        t = -R @ C

        return K, R, C

if __name__ == "__main__":
        a = (60, 0, 0, 1)
        b = (0, 0, 0, 1)
        c = (60, 60, 0, 1)
        d = (0, 60, 0, 1)
        e = (0, 105, 10, 1)
        f = (60, 105, 10, 1)
        world_points = np.array([a, b, c, d, e, f])
        A = (1862, 2570, 1)
        B = (162, 2097, 1)
        C = (1988, 1511, 1)
        D = (880, 1379, 1)
        E = (1137, 984, 1)
        F = (2049, 1022, 1)
        image_points = np.array([A, B, C, D, E, F])
        P = find_fundamental_matrix(world_points, image_points)
        K, R, C = decompose_camera_matrix(P)
        print(f"camera location is {C} cm")