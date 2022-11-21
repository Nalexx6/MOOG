import matplotlib.pyplot as plt
import json
import numpy as np
from tqdm import tqdm


def _calc_N(i, k, u):
    ti0 = values[i]
    ti1 = values[i + 1]
    if k == 1:
        return int(ti0 <= u < ti1)
    term_1 = _calc_N(i, k - 1, u) * (u - ti0) / (values[i + k - 1] - ti0)
    
    term_2 = _calc_N(i + 1, k - 1, u) * (values[i + k] - u) / (values[i + k] - ti1)
    return term_1 + term_2


def calc_N(i, u):
    return _calc_N(i, DEGREE, u)


def calc_R(i, u):
    res = calc_N(i, u)
    if res:
        res = res / sum(calc_N(j, u) for j in range(len(control_points)))
    return res


def calc_point(u):
    result = np.zeros(2)
    for k in range(len(control_points)):
        result += calc_R(k, u) * control_points[k]
    return result


if __name__ == "__main__":
    data = json.load(open('data_14.json'))

    input_points = np.array(data['curve'])
    control_points = np.array(input_points)

    DEGREE = 5
    values = (range(len(control_points) + DEGREE + 1))

    result_points = []

    for u in tqdm(np.arange(0, max(values), step=0.1)):
        points = calc_point(u)
        result_points.append(points)
    result_points = np.array(result_points)
    m = result_points.astype(bool)

    result_points = result_points[m[:, 0] & m[:, 1]]

    plt.plot(control_points[:, 0], control_points[:, 1], ':')
    plt.plot(result_points[:, 0], result_points[:, 1])
    plt.scatter(input_points[:, 0], input_points[:, 1], c='r')

    plt.savefig("result1.jpg")
