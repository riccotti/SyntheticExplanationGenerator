import numpy as np
import matplotlib.pyplot as plt


def generate_img(img_size=(32, 32, 3), cell_size=(4, 4), min_nbr_cells=0.1, max_nbr_cells=0.3,
                 colors_p=np.array([0.15, 0.7, 0.15])):
    cells_per_edge = (img_size[0] // cell_size[0], img_size[1] // cell_size[1])
    indexes = [(i, j) for i in range(cells_per_edge[0]) for j in range(cells_per_edge[1])]
    nbr_cells = len(indexes)
    min_nbr_cells = int(nbr_cells * min_nbr_cells)
    max_nbr_cells = int(nbr_cells * max_nbr_cells)

    nbr_selected_cells = np.random.randint(min_nbr_cells, max_nbr_cells)
    selected_cells = np.random.choice(range(nbr_cells), nbr_selected_cells)

    colors = {'r': np.array([1.0, 0.0, 0.0]),
              'g': np.array([0.0, 1.0, 0.0]),
              'b': np.array([0.0, 0.0, 1.0]),
              }

    img = np.zeros(img_size)
    for cell_idx in selected_cells:
        idx = indexes[cell_idx]
        rnd_color = np.random.choice(list(colors.keys()), p=colors_p)
        val = colors[rnd_color]

        i_from = idx[0] * cell_size[0]
        i_to = i_from + cell_size[0]

        j_from = idx[1] * cell_size[1]
        j_to = j_from + cell_size[1]

        for i in range(i_from, i_to):
            for j in range(j_from, j_to):
                img[i][j] = val

    return img


def generate_pattern(pattern_size_rows=16, pattern_size_cols=16, cell_size=(4, 4), p_border=0.7):

    img = generate_img(img_size=(pattern_size_rows, pattern_size_cols, 3),
                       cell_size=cell_size, min_nbr_cells=0.6, max_nbr_cells=0.9,
                       colors_p=np.array([0.15, 0.7, 0.15]))

    if np.random.random() < p_border:
        img[0:cell_size[0], :] = np.array([0.0, 0.0, 0.0])
    if np.random.random() < p_border:
        img[:, 0:cell_size[1]] = np.array([0.0, 0.0, 0.0])
    if np.random.random() < p_border:
        img[img.shape[0] - cell_size[0]:img.shape[0], :] = np.array([0.0, 0.0, 0.0])
    if np.random.random() < p_border:
        img[:, img.shape[1] - cell_size[1]:img.shape[1]] = np.array([0.0, 0.0, 0.0])

    return img


def _predict(x, p, cs):
    for i in range(0, x.shape[0] - p.shape[0] + cs[0], cs[0]):
        for j in range(0, x.shape[1] - p.shape[1] + cs[1], cs[1]):
            i1 = i + p.shape[0]
            j1 = j + p.shape[1]
            if np.array_equal(x[i:i1, j:j1], p):
                return 1.0
    return 0.0


def _predict_index(x, p, cs):
    for i in range(0, x.shape[0] - p.shape[0] + cs[0], cs[0]):
        for j in range(0, x.shape[1] - p.shape[1] + cs[1], cs[1]):
            i1 = i + p.shape[0]
            j1 = j + p.shape[1]
            if np.array_equal(x[i:i1, j:j1], p):
                return [i, i1, j, j1]
    return None


def generate_synthetic_image_classifier(img_size=(32, 32, 3), cell_size=(4, 4), n_features=(16, 16), p_border=0.7,
                                        random_state=None):
    if random_state:
        np.random.seed(random_state)

    pattern_size_rows, pattern_size_cols = n_features
    pattern = generate_pattern(pattern_size_rows, pattern_size_cols, cell_size, p_border)
    while np.sum(pattern) == 0:
        pattern = generate_pattern(pattern_size_rows, pattern_size_cols, cell_size, p_border)

    def predict_proba(X):
        proba = list()
        for x in X:
            if x.shape != img_size:
                x = x.reshape(img_size)
            val =_predict(x, pattern, cell_size)
            proba.append(np.array([1.0 - val, val]))
        proba = np.array(proba)
        return proba

    def predict(X):
        proba = predict_proba(X)
        return np.argmax(proba, axis=1)

    srbc = {
        'img_size': img_size,
        'cell_size': cell_size,
        'pattern': pattern,
        'predict_proba': predict_proba,
        'predict': predict
    }

    return srbc


def generate_random_img_dataset(pattern, nbr_images=1000, pattern_ratio=0.5, img_size=(32, 32, 3), cell_size=(4, 4),
                                min_nbr_cells=0.1, max_nbr_cells=0.3, colors_p=np.array([0.15, 0.7, 0.15])):

    X_test = list()

    nbr_images0 = int(nbr_images * (1 - pattern_ratio))
    for _ in range(nbr_images0):
        img = generate_img(img_size, cell_size, min_nbr_cells, max_nbr_cells, colors_p)
        X_test.append(img)

    nbr_images1 = int(nbr_images * pattern_ratio)
    for _ in range(nbr_images1):
        img = generate_img(img_size, cell_size, min_nbr_cells, max_nbr_cells, colors_p)
        i = np.random.choice(np.arange(0, img_size[0] + cell_size[0] - pattern.shape[0], cell_size[0]))
        j = np.random.choice(np.arange(0, img_size[1] + cell_size[1] - pattern.shape[1], cell_size[1]))
        img[i:i+pattern.shape[0], j:j+pattern.shape[1]] = pattern
        X_test.append(img)

    X_test = np.array(X_test)
    # np.random.shuffle(X_test)

    return X_test


def get_pixel_importance_explanation(x, sic):

    p = sic['pattern']
    cs = sic['cell_size']
    img_size = sic['img_size']

    explanation = np.zeros(img_size[:2])
    index = _predict_index(x, p, cs)

    if index is not None:
        i0, i1, j0, j1 = index
        for i in range(i0, i1):
            for j in range(j0, j1):
                if np.sum(x[i][j]) > 0:
                    explanation[i][j] = 1.0

    explanation = explanation.ravel()
    return explanation


def main():

    n_features = (16, 16)
    img_size = (32, 32, 3)
    cell_size = (4, 4)
    colors_p = np.array([0.15, 0.7, 0.15])

    sic = generate_synthetic_image_classifier(img_size=img_size, cell_size=cell_size, n_features=n_features)

    pattern = sic['pattern']
    predict = sic['predict']
    predict_proba = sic['predict_proba']

    plt.imshow(pattern)
    plt.show()

    X_test = generate_random_img_dataset(pattern, nbr_images=1000, pattern_ratio=0.4, img_size=img_size,
                                         cell_size=cell_size, min_nbr_cells=0.1, max_nbr_cells=0.3, colors_p=colors_p)

    Y_test = predict(X_test)

    # plt.imshow(X_test[0])
    # plt.show()

    plt.imshow(X_test[-1])
    plt.show()

    print(Y_test[-1])
    print(list(get_pixel_importance_explanation(X_test[-1], sic)))

    print(Y_test[0])
    print(list(get_pixel_importance_explanation(X_test[0], sic)))

    # for x in X_test:
    #     expl_val = get_pixel_importance_explanation(x, sic)
    #     print(expl_val)
    #     break


if __name__ == "__main__":
    main()

