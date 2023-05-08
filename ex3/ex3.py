import numpy as np
from sympy import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from functools import reduce
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def get_conic_value(case, x, y):
    if case == 1:
        val = 66051945006480660*x**2 + 33653211901387302*x*y - 17811423300513194802*x + 116495460745496226*y**2 - 418562018934895221600*y + 159380879831564658173214
    elif case == 2:
        val = 120208448259898680 * x ** 2 + 37067933567014152 * x * y - 371660175649798691832 * x + 1909278070183440 * y ** 2 - 54246393272339180880 * y + 281439625669671536168640
    elif case == 3:
        val = -151522235553106704*x**2 + 107314687356800088*x*y + 106033433046068744568*x - 111572529633243360*y**2 + 209690470146049894320*y - 111747240006289762625520
    elif case == 4:
        val = -145360750836929100 * x ** 2 + 166796075354605554 * x * y + 175229264169735980226 * x + 35860223361747774 * y ** 2 - 283320806860386217032 * y + 45183859650753728898378
    elif case == 5:
        val = -274911535827865926 * x ** 2 + 118587394202393391 * x * y + 415571159367126766179 * x + 28637546498758857 * y ** 2 - 225725493251191069092 * y - 47271697283734678999629
    return val

def find_epipole():
    # points in image 1
    p1 = np.array([1156, 2090, 1])
    p2 = np.array([1351, 2996, 1])
    p3 = np.array([1721, 2635, 1])
    p4 = np.array([1953, 1242, 1])
    p5 = np.array([1758, 2899, 1])
    e = np.array([1494, 1480, 1])

    # points in image 2
    p1_t = np.array([1510, 1331, 1])
    p2_t = np.array([1390, 2139, 1])
    p3_t = np.array([941, 2676, 1])
    p4_t = np.array([407, 465, 1])
    p5_t = np.array([1030, 2207, 1])

    all_points_orig = np.array([p1_t[:2], p2_t[:2], p3_t[:2], p4_t[:2], p5_t[:2]])

    # the true
    e_t_true = np.array([1233, 855, 1])
    points_a, cone_a_x, cone_a_y = find_cone_pixels_4_pnts_and_ep(e, p1, p1_t, p2, p2_t, p3, p3_t, p4, p4_t, case=1, thresh=1e21)
    # points_b, cone_b_x, cone_b_y = find_cone_pixels_4_pnts_and_ep(e, p1, p1_t, p2, p2_t, p3, p3_t, p5, p5_t, case=2, thresh=1e20)
    points_c, cone_c_x, cone_c_y = find_cone_pixels_4_pnts_and_ep(e, p1, p1_t, p2, p2_t, p4, p4_t, p5, p5_t, case=3, thresh=1e21)
    # points_d, cone_d_x, cone_d_y = find_cone_pixels_4_pnts_and_ep(e, p1, p1_t, p3, p3_t, p4, p4_t, p5, p5_t, case=4, thresh=1e21)
    # points_e, cone_e_x, cone_e_y = find_cone_pixels_4_pnts_and_ep(e, p2, p2_t, p3, p3_t, p4, p4_t, p5, p5_t, case=5, thresh=1e21)

    intersection = set(points_a) & set(points_c)
    in_x, in_y = zip(*intersection)
    in_x = np.array(in_x)
    in_y = np.array(in_y)
    points = np.zeros((len(in_x), 2))
    points[:, 0] = in_x
    points[:, 1] = in_y
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(points)

    centroids = kmeans.cluster_centers_
    points_orig = np.array([p1_t[:2], p2_t[:2], p4_t[:2]])

    distances = cdist(points_orig, centroids)
    closest_clusters = np.argmin(distances, axis=1)

    unassigned_cluster = np.setdiff1d(np.arange(len(centroids)), closest_clusters)[0]

    computed_epipole = centroids[unassigned_cluster]
    ep_y = computed_epipole[1]
    ep_x = computed_epipole[0]
    img = mpimg.imread("img2 with points and epipole.png")
    plt.imshow(img)
    plt.scatter(cone_a_x, cone_a_y, s=1)
    plt.scatter(cone_c_x, cone_c_y, s=1)

    # plt.scatter(cone_d_x, cone_d_y, s=2)
    # plt.scatter(cone_e_x, cone_e_y, s=2)

    plt.scatter(points_orig[:, 0], points_orig[:, 1], marker='x', color='blue',  s=200)
    plt.scatter(ep_x, ep_y, marker='x', color='red', s=200)
    plt.show()
    pass


def find_cone_pixels_4_pnts_and_ep(e, p1, p1_t, p2, p2_t, p3, p3_t, p4, p4_t, case, thresh):
    simplified_expr = extract_conic_equation(e, p1, p1_t, p2, p2_t, p3, p3_t, p4, p4_t)
    img = mpimg.imread("img2.jpg")
    array = np.zeros((img.shape[0], img.shape[1]))
    on_the_line_x = []
    on_the_line_y = []
    points = []
    for y in range(array.shape[0]):
        print(y)
        for x in range(array.shape[1]):
            val = get_conic_value(case, x, y)
            # val = simplified_expr.subs([(ex, x), (ey, y)])
            array[y, x] = val
            if np.abs(val) <= thresh:
                on_the_line_x.append(x)
                on_the_line_y.append(y)
                points.append((x, y))
    return points, on_the_line_x, on_the_line_y


def extract_conic_equation(e, p1, p1_t, p2, p2_t, p3, p3_t, p4, p4_t):
    ex, ey = symbols('ex ey')  # defining e' = (ex, ey)
    e_t = np.array([ex, ey, 1])
    M12 = Matrix([e, p1, p2]).transpose()
    M34 = Matrix([e, p3, p4]).transpose()
    M13 = Matrix([e, p1, p3]).transpose()
    M24 = Matrix([e, p2, p4]).transpose()
    M12_t = Matrix([e_t, p1_t, p2_t]).transpose()
    M34_t = Matrix([e_t, p3_t, p4_t]).transpose()
    M13_t = Matrix([e_t, p1_t, p3_t]).transpose()
    M24_t = Matrix([e_t, p2_t, p4_t]).transpose()
    expr = M12.det() * M34.det() * M13_t.det() * M24_t.det() - M12_t.det() * M34_t.det() * M13.det() * M24.det()
    simplified_expr = simplify(simplify(simplify(expr)))
    return simplified_expr


if __name__ == "__main__":
    find_epipole()






