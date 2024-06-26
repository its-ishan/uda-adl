import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
import torch
import numpy as np
import random
import os
from tqdm import tqdm
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt

unloader = transforms.ToPILImage()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def tensor_to_PIL(tensor, is_trans=False):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


def evaluate(pred, gt):
    if isinstance(pred, (list, tuple)):
        pred = pred[-1]

    pred_binary = (pred >= 0.5).float()
    pred_binary_inverse = (pred_binary == 0).float()

    gt_binary = (gt >= 0.5).float()
    gt_binary_inverse = (gt_binary == 0).float()

    TP = pred_binary.mul(gt_binary).sum()
    FP = pred_binary.mul(gt_binary_inverse).sum()
    TN = pred_binary_inverse.mul(gt_binary_inverse).sum()
    FN = pred_binary_inverse.mul(gt_binary).sum()

    if TP.item() == 0:
        TP = torch.Tensor([1]).cuda()

    # recall
    Recall = TP / (TP + FN)

    # Specificity or true negative rate
    Specificity = TN / (TN + FP)

    # Precision or positive predictive value
    Precision = TP / (TP + FP)

    # F1 score = Dice
    Dice = 2 * Precision * Recall / (Precision + Recall)

    # F2 score
    # F2 = 5 * Precision * Recall / (4 * Precision + Recall)

    # Overall accuracy
    ACC_overall = (TP + TN) / (TP + FP + FN + TN)

    # IoU for poly
    IoU_poly = TP / (TP + FP + FN)

    # IoU for background
    IoU_bg = TN / (TN + FP + FN)

    # mean IoU
    IoU_mean = (IoU_poly + IoU_bg) / 2.0

    return Recall, Specificity, Dice, ACC_overall, IoU_poly, IoU_bg, IoU_mean


class Metrics(object):
    def __init__(self, metrics_list):
        self.metrics = {}
        for metric in metrics_list:
            self.metrics[metric] = 0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            assert (k in self.metrics.keys()), "The k {} is not in metrics".format(k)
            if isinstance(v, torch.Tensor):
                v = v.item()

            self.metrics[k] += v

    def mean(self, total):
        mean_metrics = {}
        for k, v in self.metrics.items():
            mean_metrics[k] = v / total
        return mean_metrics


def test(model, dataloader, args, adabn_loader=False):
    metrics = Metrics(['recall', 'specificity', 'Dice',
                       'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean'])

    if args.is_adabn:
        # for m in model.modules():
        #     classname = m.__class__.__name__
        #     if classname.find('BatchNorm') != -1:
        #         print(m.running_mean)
        #         m.reset_running_stats()
        # model.train()
        # with torch.no_grad():
        #     for j, batch in enumerate(adabn_loader):
        #         img, gt = batch[0]['image'], batch[0]['label']

        #         img = Variable(img).cuda()
        #         output = model(img)
        model.train()
    else:
        model.eval()
    with torch.no_grad():
        total_batch = len(dataloader.dataset)
        bar = tqdm(enumerate(dataloader), total=total_batch)
        for i, data in bar:
            img, gt = data[0]['image'], data[0]['label']

            img = Variable(img).cuda()
            gt = Variable(gt).cuda()
            gt = gt.unsqueeze(1)

            _, _, _, output = model(img, img)
            if args.save_result:
                prediction = (output >= 0.5).float()
                prediction = tensor_to_PIL(prediction)
                name = data[1][0].split('/')[-1]
                prediction.save(os.path.join('/home/cyang/MPA-DA/result', name))
            _recall, _specificity, _Dice, \
                _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean = evaluate(
                output, gt)

            metrics.update(recall=_recall, specificity=_specificity,
                           Dice=_Dice, ACC_overall=_ACC_overall, IoU_poly=_IoU_poly,
                           IoU_bg=_IoU_bg, IoU_mean=_IoU_mean
                           )

    metrics_result = metrics.mean(total_batch)

    # print("Test Result on target fold{}:".format(args.target_test[-5]))
    print("Test Result:")
    print('recall: %.5f, specificity: %.5f, Dice: %.5f, '
          'ACC_overall: %.5f, IoU_poly: %.5f, IoU_bg: %.5f, IoU_mean: %.5f'
          % (metrics_result['recall'], metrics_result['specificity'],
             metrics_result['Dice'], metrics_result['ACC_overall'],
             metrics_result['IoU_poly'], metrics_result['IoU_bg'], metrics_result['IoU_mean']))
    print()
    return metrics_result['Dice']


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
