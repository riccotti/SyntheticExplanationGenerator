import numpy as np
import matplotlib.pyplot as plt


from isyege import generate_synthetic_image_classifier
from isyege import generate_random_img_dataset
from isyege import get_pixel_importance_explanation


def main():
    n_features = (20, 20)
    img_size = (32, 32, 3)
    cell_size = (4, 4)
    colors_p = np.array([0.15, 0.7, 0.15])
    p_border = 1.0

    sic = generate_synthetic_image_classifier(img_size=img_size, cell_size=cell_size, n_features=n_features,
                                              p_border=p_border)

    pattern = sic['pattern']
    predict = sic['predict']
    predict_proba = sic['predict_proba']

    plt.imshow(pattern)
    plt.xticks(())
    plt.yticks(())
    plt.savefig('../fig/pattern.png', format='png', bbox_inches='tight')
    plt.show()

    X_test = generate_random_img_dataset(pattern, nbr_images=1000, pattern_ratio=0.4, img_size=img_size,
                                         cell_size=cell_size, min_nbr_cells=0.1, max_nbr_cells=0.3, colors_p=colors_p)

    Y_test = predict(X_test)
    idx = np.where(Y_test == 1)[0][0]

    x = X_test[idx]
    plt.imshow(x)
    plt.xticks(())
    plt.yticks(())
    plt.savefig('../fig/image.png', format='png', bbox_inches='tight')
    plt.show()

    gt_val = get_pixel_importance_explanation(x, sic)
    max_val = np.nanpercentile(np.abs(gt_val), 99.9)
    plt.imshow(np.reshape(gt_val, img_size[:2]), cmap='RdYlBu', vmin=-max_val, vmax=max_val, alpha=0.7)
    plt.xticks(())
    plt.yticks(())
    plt.savefig('../fig/saliencymap.png', format='png', bbox_inches='tight')
    plt.show()

    # plt.imshow(x)
    # plt.imshow(np.reshape(gt_val, img_size[:2]), cmap='RdYlBu', vmin=-max_val, vmax=max_val, alpha=0.7)
    # plt.xticks(())
    # plt.yticks(())
    # plt.savefig('../fig/saliencymap2.png', format='png', bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":
    main()

