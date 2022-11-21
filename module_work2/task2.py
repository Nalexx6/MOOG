import matplotlib.pyplot as plt
from math import factorial as fac
import json
import numpy as np
from tqdm import tqdm
import matplotlib.tri as mpltri

GRID_SIZE = 13


def C(i, m):
    return fac(m) / (fac(i) * fac(m - i))


def bernshtein_polinom(m, i, t):
    return C(i, m) * (t ** i) * ((1 - t) ** (m - i))


def compute_point(u, v):
    coef = np.zeros(3)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            coef += bernshtein_polinom(GRID_SIZE - 1, i, u) * bernshtein_polinom(GRID_SIZE - 1, j, v) * points[indices.index([j, i])]
    return coef


if __name__ == "__main__":
    data = json.load(open('data_14.json'))
    points = np.array(data['surface']['points'])
    indices = (data['surface']['indices'])

    values = np.arange(1, step=0.05)

    result = {}
    result_points = []
    for u in tqdm(values):
        for v in values:
            result.setdefault(u, {})
            point = compute_point(u, v)
            result_points.append(point)
            result[u][v] = point
    result_points = np.array(result_points)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    result = result_points

    triangulation = mpltri.Triangulation(result[:, 0], result[:, 1])
    ax.plot_trisurf(triangulation, result[:, 2])
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r')

    fig.savefig('result2.jpg')
