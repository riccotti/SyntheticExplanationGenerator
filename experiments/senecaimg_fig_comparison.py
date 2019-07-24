import numpy as np
import matplotlib.pyplot as plt


from isyege import generate_synthetic_image_classifier
from isyege import generate_random_img_dataset, generate_img_defined
from isyege import get_pixel_importance_explanation
from evaluation import pixel_based_similarity

from MAPLE import MAPLE
from shap import KernelExplainer
from lime_image import LimeImageExplainer
from scikit_image import SegmentationAlgorithm


def main():
    n_features = (20, 20)
    img_size = (32, 32, 3)
    cell_size = (4, 4)
    colors_p = np.array([0.15, 0.7, 0.15])
    p_border = 1.0

    # img_draft = np.array([ # ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k'],
    #     ['k', 'k', 'k', 'k', 'k', 'g', 'r', 'k'],
    #     ['g', 'k', 'k', 'k', 'k', 'k', 'k', 'g'],
    #     ['k', 'g', 'k', 'k', 'k', 'b', 'k', 'k'],
    #     ['k', 'g', 'k', 'k', 'g', 'g', 'k', 'b'],
    #     ['k', 'k', 'k', 'k', 'g', 'k', 'k', 'g'],
    #     ['g', 'k', 'k', 'k', 'k', 'k', 'k', 'k'],
    #     ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k'],
    #     ['k', 'k', 'k', 'k', 'g', 'k', 'k', 'k'],
    #
    # ])
    # img = generate_img_defined(img_draft, img_size=img_size, cell_size=cell_size)
    # plt.imshow(img)
    # plt.xticks(())
    # plt.yticks(())
    # # plt.savefig('../fig/pattern.png', format='png', bbox_inches='tight')
    # plt.show()

    pattern_draft = np.array([  # ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k'],
        ['k', 'k', 'k', 'k', 'k'],
        ['k', 'k', 'k', 'b', 'k'],
        ['k', 'k', 'g', 'g', 'k'],
        ['k', 'k', 'g', 'k', 'k'],
        ['k', 'k', 'k', 'k', 'k'],

    ])

    pattern = generate_img_defined(pattern_draft, img_size=(20, 20, 3), cell_size=cell_size)

    sic = generate_synthetic_image_classifier(img_size=img_size, cell_size=cell_size, n_features=n_features,
                                              p_border=p_border, pattern=pattern)

    pattern = sic['pattern']
    predict = sic['predict']
    predict_proba = sic['predict_proba']

    plt.imshow(pattern)
    plt.xticks(())
    plt.yticks(())
    # plt.savefig('../fig/pattern.png', format='png', bbox_inches='tight')
    plt.show()

    X_test = generate_random_img_dataset(pattern, nbr_images=1000, pattern_ratio=0.4, img_size=img_size,
                                         cell_size=cell_size, min_nbr_cells=0.1, max_nbr_cells=0.3, colors_p=colors_p)

    Y_test = predict(X_test)
    idx = np.where(Y_test == 1)[0][0]

    # x = X_test[idx]
    img_draft = np.array([ # ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k'],
        ['k', 'k', 'k', 'k', 'k', 'g', 'r', 'k'],
        ['g', 'k', 'k', 'k', 'k', 'k', 'k', 'g'],
        ['k', 'g', 'k', 'k', 'k', 'b', 'k', 'k'],
        ['k', 'g', 'k', 'k', 'g', 'g', 'k', 'b'],
        ['k', 'k', 'k', 'k', 'g', 'k', 'k', 'g'],
        ['g', 'k', 'k', 'k', 'k', 'k', 'k', 'k'],
        ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k'],
        ['k', 'k', 'k', 'k', 'g', 'k', 'k', 'k'],

    ])
    x = generate_img_defined(img_draft, img_size=img_size, cell_size=cell_size)
    plt.imshow(x)
    plt.xticks(())
    plt.yticks(())
    # plt.savefig('../fig/image.png', format='png', bbox_inches='tight')
    plt.show()

    gt_val = get_pixel_importance_explanation(x, sic)
    max_val = np.nanpercentile(np.abs(gt_val), 99.9)
    plt.imshow(np.reshape(gt_val, img_size[:2]), cmap='RdYlBu', vmin=-max_val, vmax=max_val, alpha=0.7)
    plt.xticks(())
    plt.yticks(())
    # plt.savefig('../fig/saliencymap.png', format='png', bbox_inches='tight')
    plt.show()

    # plt.imshow(x)
    # plt.imshow(np.reshape(gt_val, img_size[:2]), cmap='RdYlBu', vmin=-max_val, vmax=max_val, alpha=0.7)
    # plt.xticks(())
    # plt.yticks(())
    # plt.savefig('../fig/saliencymap2.png', format='png', bbox_inches='tight')
    # plt.show()

    lime_explainer = LimeImageExplainer()
    segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=10, ratio=0.5)
    tot_num_features = img_size[0] * img_size[1]

    lime_exp = lime_explainer.explain_instance(x, predict_proba, top_labels=2, hide_color=0,
                                               num_samples=10000, segmentation_fn=segmenter)
    _, lime_expl_val = lime_exp.get_image_and_mask(1, positive_only=True, num_features=tot_num_features,
                                                   hide_rest=False, min_weight=0.0)
    max_val = np.nanpercentile(np.abs(lime_expl_val), 99.9)
    plt.imshow(lime_expl_val, cmap='RdYlBu', vmin=-max_val, vmax=max_val, alpha=0.7)
    plt.xticks(())
    plt.yticks(())
    plt.title('lime', fontsize=20)
    plt.savefig('../fig/saliencymap_lime.png', format='png', bbox_inches='tight')
    plt.show()

    background = np.array([np.zeros(img_size).ravel()] * 10)
    shap_explainer = KernelExplainer(predict_proba, background)

    shap_expl_val = shap_explainer.shap_values(x.ravel(), l1_reg='bic')[1]
    shap_expl_val = np.sum(np.reshape(shap_expl_val, img_size), axis=2)
    tmp = np.zeros(shap_expl_val.shape)
    tmp[np.where(shap_expl_val > 0.0)] = 1.0
    shap_expl_val = tmp
    max_val = np.nanpercentile(np.abs(shap_expl_val), 99.9)
    plt.imshow(shap_expl_val, cmap='RdYlBu', vmin=-max_val, vmax=max_val, alpha=0.7)
    plt.xticks(())
    plt.yticks(())
    plt.title('shap', fontsize=20)
    plt.savefig('../fig/saliencymap_shap.png', format='png', bbox_inches='tight')
    plt.show()


    nbr_records = 10
    Xm_test = np.array([x.ravel() for x in X_test[:nbr_records]])
    maple_explainer = MAPLE(Xm_test, Y_test[:nbr_records], Xm_test, Y_test[:nbr_records],
                            n_estimators=5, max_features=0.5, min_samples_leaf=5)

    maple_exp = maple_explainer.explain(x)
    maple_expl_val = maple_exp['coefs'][:-1]
    maple_expl_val = np.sum(np.reshape(maple_expl_val, img_size), axis=2)
    tmp = np.zeros(maple_expl_val.shape)
    tmp[np.where(maple_expl_val > 0.0)] = 1.0
    maple_expl_val = tmp
    max_val = np.nanpercentile(np.abs(shap_expl_val), 99.9)
    plt.imshow(maple_expl_val, cmap='RdYlBu', vmin=-max_val, vmax=max_val, alpha=0.7)
    plt.xticks(())
    plt.yticks(())
    plt.title('maple', fontsize=20)
    plt.savefig('../fig/saliencymap_maple.png', format='png', bbox_inches='tight')
    plt.show()

    lime_f1, lime_pre, lime_rec = pixel_based_similarity(lime_expl_val.ravel(), gt_val, ret_pre_rec=True)
    shap_f1, shap_pre, shap_rec = pixel_based_similarity(shap_expl_val.ravel(), gt_val, ret_pre_rec=True)
    maple_f1, maple_pre, maple_rec = pixel_based_similarity(maple_expl_val.ravel(), gt_val, ret_pre_rec=True)

    print(lime_f1, lime_pre, lime_rec)
    print(shap_f1, shap_pre, shap_rec)
    print(maple_f1, maple_pre, maple_rec)


if __name__ == "__main__":
    main()

