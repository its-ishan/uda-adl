import numpy as np
from PIL import Image
from sklearn.manifold import TSNE

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif", ".gif", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath)
    img = img.resize((512, 512), Image.BICUBIC)
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.astype(np.uint8)
#    print(image_numpy.shape)
    image_pil = Image.fromarray(np.squeeze(image_numpy))
    image_pil.save(filename)
#    print "Image saved as {}".format(filename)



# Function to compute t-SNE on a batch of features
def compute_tsne(features_batch):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(features_batch)
    return tsne_result

# Function to plot t-SNE results
def plot_tsne(tsne_results, labels):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10')
    plt.title('t-SNE of Extracted Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.colorbar()
    plt.show()
