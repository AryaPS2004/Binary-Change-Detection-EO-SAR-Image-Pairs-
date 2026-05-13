import matplotlib.pyplot as plt

def visualize_prediction(pre, post, mask, pred, save_path):

    fig, ax = plt.subplots(1, 4, figsize=(16, 5))

    ax[0].imshow(pre)
    ax[0].set_title("Pre")

    ax[1].imshow(post)
    ax[1].set_title("Post")

    ax[2].imshow(mask, cmap="gray")
    ax[2].set_title("GT")

    ax[3].imshow(pred, cmap="gray")
    ax[3].set_title("Prediction")

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()