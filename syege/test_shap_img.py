import numpy as np
import matplotlib.pyplot as plt

from shap import KernelExplainer

from isyege import generate_synthetic_image_classifier
from isyege import generate_random_img_dataset
from isyege import get_pixel_importance_explanation
from evaluation import pixel_based_similarity


def main():
    n_features = (8, 8)
    img_size = (32, 32, 3)
    cell_size = (4, 4)
    colors_p = np.array([0.15, 0.7, 0.15])
    p_border = 0.0

    sic = generate_synthetic_image_classifier(img_size=img_size, cell_size=cell_size, n_features=n_features,
                                              p_border=p_border)

    pattern = sic['pattern']
    predict = sic['predict']
    predict_proba = sic['predict_proba']

    plt.imshow(pattern)
    plt.show()

    X_test = generate_random_img_dataset(pattern, nbr_images=1000, pattern_ratio=0.4, img_size=img_size,
                                         cell_size=cell_size, min_nbr_cells=0.1, max_nbr_cells=0.3, colors_p=colors_p)

    Y_test = predict(X_test)

    background = np.array([np.zeros(img_size).ravel()] * 10)
    explainer = KernelExplainer(predict_proba, background)

    x = X_test[-1]
    plt.imshow(x)
    plt.show()
    print(Y_test[-1])
    expl_val = explainer.shap_values(x.ravel())[1]
    # expl_val = (expl_val - np.min(expl_val)) / (np.max(expl_val) - np.min(expl_val))
    print(expl_val)
    print(np.unique(expl_val, return_counts=True))
    print(expl_val.shape)

    sv = np.sum(np.reshape(expl_val, img_size), axis=2)
    sv01 = np.zeros(sv.shape)
    sv01[np.where(sv > 0.0)] = 1.0
    # np.array([1.0 if v > 0.0 else 0.0 for v in expl_val])
    sv = sv01
    print(sv)
    print(sv.shape)

    max_val = np.nanpercentile(np.abs(sv), 99.9)
    # plt.imshow(x)
    plt.imshow(sv, cmap='RdYlBu', vmin=-max_val, vmax=max_val, alpha=0.7)
    plt.show()
    # shap.image_plot(expl_val, x)

    gt_val = get_pixel_importance_explanation(x, sic)
    print(gt_val.shape)
    max_val = np.nanpercentile(np.abs(gt_val), 99.9)
    # plt.imshow(x)
    plt.imshow(np.reshape(gt_val, img_size[:2]), cmap='RdYlBu', vmin=-max_val, vmax=max_val, alpha=0.7)
    plt.show()

    print(pixel_based_similarity(sv.ravel(), gt_val))

    # for i, x in enumerate(X_test):
    #     print(x)
    #     expl_val = explainer.shap_values(x)[1]
    #     gt_val = get_feature_importance_explanation(x, slc, n_features, get_values=True)
    #     fis = feature_importance_similarity(expl_val, gt_val)
    #     print(expl_val)
    #     print(gt_val)
    #     print(fis)
    #     print('')
    #     if i == 10:
    #         break


if __name__ == "__main__":
    main()

