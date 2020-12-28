import numpy as np


def lev(str_a, str_b):
    """
    ED距离，用来衡量单词之间的相似度
    :param str_a:
    :param str_b:
    :return:
    """
    str_a = str_a.lower()
    str_b = str_b.lower()
    matrix_ed = np.zeros((len(str_a)+1, len(str_b)+1), dtype=np.int)
    matrix_ed[0] = np.arange(len(str_b)+1)
    matrix_ed[:, 0] = np.arange(len(str_a) + 1)
    for i in range(1, len(str_a)+1):
        for j in range(1, len(str_b)+1):
            # 表示删除a_i
            dist_1 = matrix_ed[i - 1, j] + 1
            # 表示插入b_i
            dist_2 = matrix_ed[i, j - 1] + 1
            # 表示替换b_i
            # dist_3 = matrix_ed[i - 1, j - 1] + (1 if str_a[i - 1] != str_b[j - 1] else 0)
            dist_3 = matrix_ed[i - 1, j - 1] + (2 if str_a[i - 1] != str_b[j - 1] else 0)
            # 取最小距离
            matrix_ed[i, j] = np.min([dist_1, dist_2, dist_3])
    print(matrix_ed)
    return matrix_ed[-1, -1]