import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys


def poly_kernel(points, labels, exp, dim):
    '''
    Homebrew polynomial kernel. Take in labeled points of given dimensionality. For each point x add a dimension
    whose value is equal to the sum of y * <x, x'>**d for all pair-wise combinations. This will give us a linear
    separable data-set in higher dimensional space
    :param points: list of points given as np.arrays
    :param labels: list of labels, either 1 or -1
    :param exp: exponent for polynomial kernel
    :param dim: dimensionality of current data
    :return: list of linearly separable points in higher dimension
    '''
    k_points = np.zeros(shape=(len(points), dim+1))
    for ix, x in enumerate(points):
        k_points[ix] = np.append(x, 0)
        for iy, y in enumerate(points):
            k_points[ix, dim] += labels[iy] * (np.dot(x, y)**exp)
    return k_points


if __name__ == "__main__":

    usage = "python3 kernel_trick.py [quadratic|cubic]"
    if len(sys.argv) != 2 or (sys.argv[1] not in ['cubic','quadratic','test']):
        print(usage)
        exit(1)
    exponent = sys.argv[1]

    # Get 200, 2 dimensional points
    dim = 2
    n = 200

    points = np.random.uniform(low=-1, high=1, size=(n, dim))
    labels = np.empty(n)

    if exponent == 'quadratic':
        # Creating a set of 2-d data seperable by a random circle
        radius = 0.5
        center = np.random.uniform(low=-1, high=1, size=dim)

        # Generate labels according to points' position inside or outside of circle
        for i, v in enumerate(points):
            labels[i] = 1 if euclidean(v, center) > radius else -1

        # Transform points into higher dimensional space with quadratic polynomial kernel
        kernel_points = poly_kernel(points, labels, exp=2, dim=2)

    if exponent == 'cubic':
        # Equation y = mx + b with random m and b
        x_coef = np.random.uniform(low=-1, high = 1)
        b = np.random.uniform(low=-1, high=1)

        # Generate labels accodring to points' position on either side of cubic curve
        for i, v in enumerate(points):
            labels[i] = 1 if v[1] > (x_coef * v[0] + b) else -1

        # Transform points into higher dimensional space with cubic polynomial kernel
        kernel_points = poly_kernel(points, labels, exp=3, dim=2)


    # Run linear svm on transformed points for separation
    svm = SVC(kernel="linear")
    svm.fit(kernel_points, labels)
    w = svm.coef_[0]
    b = svm.intercept_
    print("weights = {}".format(w))

    # Give shapes and colors according to labels
    print(w)
    shapes = []
    colors = []
    for l in labels:
        if l == 1:
            shapes.append("^")
            colors.append("r")
        else:
            shapes.append("o")
            colors.append("b")

    # Make a 3d plot!
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, x in enumerate(kernel_points):
        ax.scatter(x[0], x[1], x[2], c=colors[i], marker=shapes[i])

    [xx, yy] = np.meshgrid(np.arange(-1, 1, 0.1), np.arange(-1, 1, 0.1))
    z = (- w[0] * xx - w[1] * yy - b) / w[2]
    ax.plot_surface(xx, yy, z, color='g', alpha=0.5)
    plt.show()

