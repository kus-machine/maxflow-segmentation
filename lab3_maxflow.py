import cv2
import maxflow
import numpy as np


def take_params(image, h=.2, w=.4, offset=0.05):
    if w > 0.5:
        w = 0.5
    piece0 = image[int(offset * image.shape[0]):int((h + offset) * image.shape[0]),
             int(image.shape[1] * (1 / 2 - w)):int(
                 image.shape[1] * (1 / 2 + w))]
    piece1 = image[int((1 - h - offset) * image.shape[0]):int((1 - offset) * image.shape[0]), int(image.shape[1] * (
            1 / 2 - w)):int(image.shape[1] * (1 / 2 + w))]

    piece0 = np.reshape(piece0, (piece0.shape[0] * piece0.shape[1], piece0.shape[2]))
    piece1 = np.reshape(piece1, (piece1.shape[0] * piece1.shape[1], piece1.shape[2]))

    mean_0 = np.mean(piece0, axis=0)
    mean_1 = np.mean(piece1, axis=0)

    cov_0 = np.cov([piece0[..., 0], piece0[..., 1], piece0[..., 2]])
    cov_1 = np.cov([piece1[..., 0], piece1[..., 1], piece1[..., 2]])

    return mean_0, mean_1, cov_0, cov_1


def pdf_multivariate_gauss_log(x, mu, cov):
    cov = cov.astype(float)
    part1 = np.log(np.linalg.det(cov))
    part2 = (-1 / 2) * ((x - mu).T.dot(np.linalg.inv(cov))).dot((x - mu))
    return part2 - 2 * part1


def get_probs(image, mean_0, mean_1, cov_0, cov_1):
    image_resh = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
    PROB = np.zeros((image_resh.shape[0], 2), float)
    for i in range(image_resh.shape[0]):
        # for j in prange(image.shape[1]-1):
        PROB[i, 0] = pdf_multivariate_gauss_log(image_resh[i], mean_0, cov_0)
        PROB[i, 1] = pdf_multivariate_gauss_log(image_resh[i], mean_1, cov_1)
    return PROB.reshape((image.shape[0], image.shape[1], 2))


def n_of_Nt(i, j, shape):
    n = 4
    if i == 0:
        n -= 1
    if i == (shape[0] - 1):
        n -= 1
    if j == 0:
        n -= 1
    if j == (shape[1] - 1):
        n -= 1
    return n


def try_build_labeling(gr, labels):
    image_is_good = True
    labels[..., :] = 0
    for i, j in np.ndindex(gr.shape[:2]):
        if image_is_good:
            if gr[i, j].max() == -np.inf:
                image_is_good = False
                return image_is_good, labels
    for i, j in np.ndindex(labels.shape[:2]):
        if i == 0:
            if j == 0:
                class0 = max(gr[2 * i + 1, 2 * j, 0], gr[2 * i + 1, 2 * j, 1], gr[2 * i, 2 * j + 1, 0],
                             gr[2 * i, 2 * j + 1, 1])
                class1 = max(gr[2 * i + 1, 2 * j, 2], gr[2 * i + 1, 2 * j, 3], gr[2 * i, 2 * j + 1, 2],
                             gr[2 * i, 2 * j + 1, 3])
                if class1 > class0:
                    labels[i, j] = 1
            elif j == labels.shape[1] - 1:
                class0 = max(gr[2 * i + 1, 2 * j, 0], gr[2 * i + 1, 2 * j, 1], gr[2 * i, 2 * j - 1, 0],
                             gr[2 * i, 2 * j - 1, 2])
                class1 = max(gr[2 * i + 1, 2 * j, 2], gr[2 * i + 1, 2 * j, 3], gr[2 * i, 2 * j - 1, 1],
                             gr[2 * i, 2 * j - 1, 3])
                if class1 > class0:
                    labels[i, j] = 1
            else:
                class0 = max(gr[2 * i + 1, 2 * j, 0], gr[2 * i + 1, 2 * j, 1], gr[2 * i, 2 * j + 1, 0],
                             gr[2 * i, 2 * j + 1, 1], gr[2 * i, 2 * j - 1, 0], gr[2 * i, 2 * j - 1, 2])
                class1 = max(gr[2 * i + 1, 2 * j, 2], gr[2 * i + 1, 2 * j, 3], gr[2 * i, 2 * j + 1, 2],
                             gr[2 * i, 2 * j + 1, 3], gr[2 * i, 2 * j - 1, 1], gr[2 * i, 2 * j - 1, 3])
                if class1 > class0:
                    labels[i, j] = 1
        elif i == labels.shape[0] - 1:
            if j == 0:
                class0 = max(gr[2 * i, 2 * j + 1, 0], gr[2 * i, 2 * j + 1, 1], gr[2 * i - 1, 2 * j, 0],
                             gr[2 * i - 1, 2 * j, 2])
                class1 = max(gr[2 * i, 2 * j + 1, 2], gr[2 * i, 2 * j + 1, 3], gr[2 * i - 1, 2 * j, 1],
                             gr[2 * i - 1, 2 * j, 3])
                if class1 > class0:
                    labels[i, j] = 1
            elif j == labels.shape[1] - 1:
                class0 = max(gr[2 * i - 1, 2 * j, 0], gr[2 * i - 1, 2 * j, 2], gr[2 * i, 2 * j - 1, 0],
                             gr[2 * i, 2 * j - 1, 2])
                class1 = max(gr[2 * i - 1, 2 * j, 1], gr[2 * i - 1, 2 * j, 3], gr[2 * i, 2 * j - 1, 1],
                             gr[2 * i, 2 * j - 1, 3])
                if class1 > class0:
                    labels[i, j] = 1
            else:
                class0 = max(gr[2 * i, 2 * j + 1, 0], gr[2 * i, 2 * j + 1, 1], gr[2 * i - 1, 2 * j, 0],
                             gr[2 * i - 1, 2 * j, 2], gr[2 * i, 2 * j - 1, 0],
                             gr[2 * i, 2 * j - 1, 2])
                class1 = max(gr[2 * i, 2 * j + 1, 2], gr[2 * i, 2 * j + 1, 3], gr[2 * i - 1, 2 * j, 1],
                             gr[2 * i - 1, 2 * j, 3], gr[2 * i, 2 * j - 1, 1],
                             gr[2 * i, 2 * j - 1, 3])
                if class1 > class0:
                    labels[i, j] = 1
        else:
            if j == 0:
                class0 = max(gr[2 * i + 1, 2 * j, 0], gr[2 * i + 1, 2 * j, 1], gr[2 * i, 2 * j + 1, 0],
                             gr[2 * i, 2 * j + 1, 1], gr[2 * i - 1, 2 * j, 0], gr[2 * i - 1, 2 * j, 2])
                class1 = max(gr[2 * i + 1, 2 * j, 2], gr[2 * i + 1, 2 * j, 3], gr[2 * i, 2 * j + 1, 2],
                             gr[2 * i, 2 * j + 1, 3], gr[2 * i - 1, 2 * j, 1], gr[2 * i - 1, 2 * j, 3], )
                if class1 > class0:
                    labels[i, j] = 1
            elif j == labels.shape[1] - 1:
                class0 = max(gr[2 * i + 1, 2 * j, 0], gr[2 * i + 1, 2 * j, 1], gr[2 * i - 1, 2 * j, 0],
                             gr[2 * i - 1, 2 * j, 2], gr[2 * i, 2 * j - 1, 0], gr[2 * i, 2 * j - 1, 2])
                class1 = max(gr[2 * i + 1, 2 * j, 2], gr[2 * i + 1, 2 * j, 3], gr[2 * i - 1, 2 * j, 1],
                             gr[2 * i - 1, 2 * j, 3], gr[2 * i, 2 * j - 1, 1], gr[2 * i, 2 * j - 1, 3])
                if class1 > class0:
                    labels[i, j] = 1
            else:
                class0 = max(gr[2 * i + 1, 2 * j, 0], gr[2 * i + 1, 2 * j, 1], gr[2 * i, 2 * j + 1, 0],
                             gr[2 * i, 2 * j + 1, 1], gr[2 * i - 1, 2 * j, 0], gr[2 * i - 1, 2 * j, 2],
                             gr[2 * i, 2 * j - 1, 0], gr[2 * i, 2 * j - 1, 2])
                class1 = max(gr[2 * i, 2 * j - 1, 3], gr[2 * i - 1, 2 * j, 3], gr[2 * i - 1, 2 * j, 1],
                             gr[2 * i, 2 * j + 1, 3], gr[2 * i, 2 * j + 1, 2], gr[2 * i + 1, 2 * j, 3],
                             gr[2 * i + 1, 2 * j, 2])

                if class1 > class0:
                    labels[i, j] = 1
    return labels


def fill_graph_edges(probs, epsilon=0.25):
    g_for_im = np.zeros((2 * probs.shape[0] - 1, 2 * probs.shape[1] - 1, 4), float)
    # 4 edges - 0-0, 0-1, 1-0, 1-1 in this order
    for i in range(g_for_im.shape[0]):
        # четные i; смотрим объекты слева и справа
        if i % 2 == 0:
            for j in range(1, g_for_im.shape[1] - 1, 2):
                g_for_im[i, j, 0] = (probs[i // 2, (j - 1) // 2, 0] + probs[i // 2, (j + 1) // 2, 0]) / n_of_Nt(
                    i, j, g_for_im.shape) + 2 * np.log(epsilon)
                g_for_im[i, j, 1] = (probs[i // 2, (j - 1) // 2, 0] + probs[i // 2, (j + 1) // 2, 1]) / n_of_Nt(
                    i, j, g_for_im.shape) + 2 * np.log(1 - epsilon)
                g_for_im[i, j, 2] = (probs[i // 2, (j - 1) // 2, 1] + probs[i // 2, (j + 1) // 2, 0]) / n_of_Nt(
                    i, j, g_for_im.shape) + 2 * np.log(1 - epsilon)
                g_for_im[i, j, 3] = (probs[i // 2, (j - 1) // 2, 1] + probs[i // 2, (j + 1) // 2, 1]) / n_of_Nt(
                    i, j, g_for_im.shape) + 2 * np.log(epsilon)
        # нечетные i; смотрим объекты сверху и снизу
        else:
            for j in range(0, g_for_im.shape[1], 2):
                g_for_im[i, j, 0] = (probs[(i - 1) // 2, j // 2, 0] + probs[(i + 1) // 2, j // 2, 0]) / n_of_Nt(
                    i, j, g_for_im.shape) + 2 * np.log(epsilon)
                g_for_im[i, j, 1] = (probs[(i - 1) // 2, j // 2, 0] + probs[(i + 1) // 2, j // 2, 1]) / n_of_Nt(
                    i, j, g_for_im.shape) + 2 * np.log(1 - epsilon)
                g_for_im[i, j, 2] = (probs[(i - 1) // 2, j // 2, 1] + probs[(i + 1) // 2, j // 2, 0]) / n_of_Nt(
                    i, j, g_for_im.shape) + 2 * np.log(1 - epsilon)
                g_for_im[i, j, 3] = (probs[(i - 1) // 2, j // 2, 1] + probs[(i + 1) // 2, j // 2, 1]) / n_of_Nt(
                    i, j, g_for_im.shape) + 2 * np.log(epsilon)
    return g_for_im


def update_MFGraph(MFGraph, nodeids, custom_graph):
    for i in range(custom_graph.shape[0]):
        if i % 2 == 0:
            for j in range(0, custom_graph.shape[1], 2):
                # add g(k,k`) weights:
                # left
                if j > 2:
                    MFGraph.add_edge(nodeids[i // 2, j // 2], nodeids[i // 2, j // 2 - 2],
                                     custom_graph[i, j - 1, 1] + custom_graph[i, j - 1, 2] - custom_graph[i, j - 1, 0] -custom_graph[i, j - 1, 3], 0)
                # right
                if j < custom_graph.shape[1] - 3:
                    MFGraph.add_edge(nodeids[i // 2, j // 2], nodeids[i // 2, j // 2 + 2],
                                     custom_graph[i, j + 1, 1] + custom_graph[i, j + 1, 2] - custom_graph[i, j + 1, 0] -
                                     custom_graph[i, j + 1, 3], 0)
                # top
                if i > 2:
                    MFGraph.add_edge(nodeids[i // 2, j // 2], nodeids[i // 2 - 2, j // 2],
                                     custom_graph[i - 1, j, 1] + custom_graph[i - 1, j, 2] - custom_graph[i - 1, j, 0] -
                                     custom_graph[i - 1, j, 3], 0)
                # bot
                if i < custom_graph.shape[0] - 3:
                    MFGraph.add_edge(nodeids[i // 2, j // 2], nodeids[i // 2 + 2, j // 2],
                                     custom_graph[i + 1, j, 1] + custom_graph[i + 1, j, 2] - custom_graph[i + 1, j, 0] -
                                     custom_graph[i + 1, j, 3], 0)
    return MFGraph


# image preprocessing
filename = "field6.jpg"
img1 = cv2.imread(filename)
scale = [512, 512]
img1_res = cv2.resize(img1, scale).astype(float)

# prepare weights (from lab2)
mean0, mean1, cov0, cov1 = take_params(img1_res, h=.2, w=.4, offset=0.05)
PROBS = get_probs(img1_res, mean0, mean1, cov0, cov1)
custom_gr = fill_graph_edges(PROBS)

# build graph for maxflow alg
graph = maxflow.GraphFloat()
nodeid = graph.add_grid_nodes(img1_res.shape[:2])
graph.add_grid_tedges(nodeid, PROBS[..., 0], PROBS[..., 1])
graph = update_MFGraph(graph, nodeid, custom_gr)
flow = graph.maxflow()
maxflow_result = graph.get_grid_segments(nodeid)

# check only labels from distribution
labels = np.zeros(img1_res.shape[:2])
legacy_labels = try_build_labeling(custom_gr, labels)

cv2.imshow("win", cv2.hconcat([
    cv2.cvtColor(maxflow_result.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR),
    img1_res.astype(np.uint8),
    cv2.cvtColor(legacy_labels.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)]))
cv2.waitKey(0)
