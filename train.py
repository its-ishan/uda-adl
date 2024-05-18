from dataloader import dataloader
from configs import config_loader
from models.networks import define_G, define_D, GANLoss, print_network

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.cuda.amp import autocast, GradScaler

# Load the configuration
config = config_loader.load_config('configs/config.yaml')


# Define Logging
def init_log():
    logging.basicConfig(level=logging.ERROR,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        filename=os.path.join('logs', 'log_%d_%s.log' % (config.batch_size, config.star)),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.ERROR)
    logging.getLogger('').addHandler(console)
    return logging


logging = init_log()

if config.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without cuda in config")

cudnn.benchmark = False

torch.manual_seed(config.seed)
if config.cuda:
    torch.cuda.manual_seed(config.seed)

logging.critical('===> Loading datasets')

# Initialize DataLoader
train_set = dataloader.get_training_set(config.root_dir)
training_data_loader = DataLoader(dataset=train_set, num_workers=config.threads, batch_size=config.batch_size,
                                  shuffle=True)

# Building Model
logging.critical(f'Number of cuda devices found: {str(torch.cuda.device_count())}')
logging.critical('===> Building model')

if config.modelG:
    netG = torch.load(config.modelG)
else:
    netG = define_G(config.input_nc, config.output_nc, config.ngf, 'batch', False, range(torch.cuda.device_count()))
if config.modelD:
    netD = torch.load(config.modelD)
else:
    netD = define_D(config.input_nc + config.output_nc, config.ndf, 'batch', False, range(torch.cuda.device_count()))

# Extract features
features = []


def hook_fn(module, input, output):
    features.append(output.cpu().data.numpy())


layer = netG.model[11]
layer.register_forward_hook(hook_fn)

criterionGAN = GANLoss()
criterionL1 = nn.L1Loss()
criterionMSE = nn.MSELoss()

# Setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

logging.critical('---------- Networks initialized -------------')
print_network(netG)
print_network(netD)
logging.critical('-----------------------------------------------')

real_a = torch.FloatTensor(config.batch_size, config.input_nc, 512, 512).cuda()
real_b = torch.FloatTensor(config.batch_size, config.output_nc, 512, 512).cuda()

if config.cuda:
    netD = netD.cuda()
    netG = netG.cuda()
    criterionGAN = criterionGAN.cuda()
    criterionL1 = criterionL1.cuda()
    criterionMSE = criterionMSE.cuda()
    real_a = real_a.cuda()
    real_b = real_b.cuda()

real_a = Variable(real_a)
real_b = Variable(real_b)

all_features = []

scaler = GradScaler()

for iteration, batch in enumerate(training_data_loader, 1):
    if iteration > 100:  # Limit the number of batches for t-SNE to avoid memory issues
        break
    real_a_cpu, real_b_cpu = batch[0], batch[1]
    real_a.resize_(real_a_cpu.size()).copy_(real_a_cpu)
    real_b.resize_(real_b_cpu.size()).copy_(real_b_cpu)

    with autocast():
        outputG = netG(real_a)

    if len(features) > 0:
        extracted_features = features[0]
        all_features.append(extracted_features)
        features.clear()

# Concatenate all features
all_features = np.concatenate(all_features, axis=0)
print(f"All features shape: {all_features.shape}")

# Flatten the features for t-SNE
all_features_flattened = all_features.reshape(all_features.shape[0], -1)
print(f"All features flattened shape: {all_features_flattened.shape}")

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
tsne_results = tsne.fit_transform(all_features_flattened)

# Plot t-SNE
plt.figure(figsize=(10, 8))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=5, cmap='tab10')
plt.title('t-SNE of Extracted Features')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()
