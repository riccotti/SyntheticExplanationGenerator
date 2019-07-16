import numpy as np
import matplotlib.pyplot as plt

from lime_image import LimeImageExplainer
from scikit_image import SegmentationAlgorithm
from skimage.color import gray2rgb, rgb2gray, label2rgb

from isyege import generate_synthetic_image_classifier
from isyege import generate_random_img_dataset
from isyege import get_pixel_importance_explanation
from evaluation import pixel_based_similarity


def main():
    n_features = (16, 16)
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
    plt.show()

    X_test = generate_random_img_dataset(pattern, nbr_images=1000, pattern_ratio=0.4, img_size=img_size,
                                         cell_size=cell_size, min_nbr_cells=0.1, max_nbr_cells=0.3, colors_p=colors_p)

    Y_test = predict(X_test)
    # img = X_test[0]

    from skimage.segmentation import mark_boundaries
    # from skimage.color import rgb2gray
    # from skimage.filters import sobel
    # from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
    #
    # segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    # segments_slic = slic(img, n_segments=250, compactness=10, sigma=1)
    # segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    # gradient = sobel(rgb2gray(img))
    # segments_watershed = watershed(gradient, markers=250, compactness=0.001)
    #
    # print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))
    # print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
    # print('Quickshift number of segments: {}'.format(len(np.unique(segments_quick))))

    # fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    # ax[0, 0].imshow(mark_boundaries(img, segments_fz))
    # ax[0, 0].set_title("Felzenszwalbs's method")
    # ax[0, 1].imshow(mark_boundaries(img, segments_slic))
    # ax[0, 1].set_title('SLIC')
    # ax[1, 0].imshow(mark_boundaries(img, segments_quick))
    # plt.imshow(mark_boundaries(img, segments_quick))
    # ax[1, 0].set_title('Quickshift')
    # ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
    # ax[1, 1].set_title('Compact watershed')

    # for a in ax.ravel():
    #     a.set_axis_off()
    #
    # plt.tight_layout()
    # plt.show()


    explainer = LimeImageExplainer()
    segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=10, ratio=0.5)
    # segmenter = SegmentationAlgorithm('slic', n_segments=200, compactness=10, sigma=0, min_size_factor=10)
    # segmenter = SegmentationAlgorithm('felzenszwalb', scale=0.1, sigma=1, min_size=2)

    for x, y in zip(X_test[-1:], Y_test[-1:]):
        print(y)
        # plt.imshow(x)
        # plt.show()
        exp = explainer.explain_instance(x, predict_proba, top_labels=2, hide_color=127,
                                         num_samples=10000, segmentation_fn=segmenter)
        temp, mask = exp.get_image_and_mask(y, positive_only=True, num_features=1000, hide_rest=False,
                                                    min_weight=0.0)
        print(np.unique(temp), 'a')
        print(np.unique(mask), 'b')
        print(temp)
        print(mask)  # usare mask come feature importance

        max_val = np.nanpercentile(np.abs(mask), 99.9)
        # plt.imshow(x)
        plt.imshow(mask, cmap='RdYlBu', vmin=-max_val, vmax=max_val, alpha=0.7)
        plt.show()

        gt_val = get_pixel_importance_explanation(x, sic)
        print(gt_val.shape)
        max_val = np.nanpercentile(np.abs(gt_val), 99.9)
        # plt.imshow(x)
        plt.imshow(np.reshape(gt_val, img_size[:2]), cmap='RdYlBu', vmin=-max_val, vmax=max_val, alpha=0.7)
        plt.show()

        print(pixel_based_similarity(mask.ravel(), gt_val))

        # plt.imshow(mark_boundaries(x, mask))
        # plt.show()
        # plt.imshow(label2rgb(mask, temp, bg_label=0), interpolation='nearest')
        # # plt.imshow(label2rgb(3 - mask, temp, bg_label=255), interpolation='nearest')
        # plt.show()
        # # expl_val = {e[0]: e[1] for e in exp.as_list()}
        # gt_val = get_pixel_importance_explanation(x, sic)
        # # wbs = word_based_similarity(expl_val, gt_val, use_values=False)
        # # print(expl_val)
        # print(gt_val)
        # print(np.unique(gt_val))
        # # print(wbs, word_based_similarity(expl_val, gt_val, use_values=True))
        # print('')


if __name__ == "__main__":
    main()

