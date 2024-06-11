from dataloader import dataloader
from configs import config_loader
from models.networks import define_G, define_D, GANLoss, print_network
import util.util
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import logging
import torch.utils.data as data
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return data.default_collate(batch)

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
                                  shuffle=True, collate_fn=custom_collate)

# Building Model
logging.critical(f'Number of cuda devices found: {str(torch.cuda.device_count())}')
logging.critical('===> Building model')

if config.modelG:
    netG = torch.load(config.modelG)
else:
    netG = define_G(config.input_nc, config.output_nc, config.ngf, 'batch', False, range(torch.cuda.device_count()))

# if config.modelD:
#     netD = torch.load(config.modelD)
# else:
#     netD = define_D(config.input_nc + config.output_nc, config.ndf, 'batch', False, range(torch.cuda.device_count()))

# Extract features
features_a = []
features_b = []


def hook_fn_a(module, input, output):
    features_a.append(output.cpu().data.numpy())
def hook_fn_b(module, input, output):
    features_b.append(output.cpu().data.numpy())


# layer_a = netG.model[11]
# layer_b = netG.model[11]
# layer_a.register_forward_hook(hook_fn_a)
# layer_b.register_forward_hook(hook_fn_b)

criterionGAN = GANLoss()
criterionL1 = nn.L1Loss()
criterionMSE = nn.MSELoss()

# Setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
#optimizerD = optim.Adam(netD.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

logging.critical('---------- Networks initialized -------------')
print_network(netG)
#print_network(netD)
logging.critical('-----------------------------------------------')

real_a = torch.FloatTensor(config.batch_size, config.input_nc, 512, 512).cuda()
real_b = torch.FloatTensor(config.batch_size, config.output_nc, 512, 512).cuda()

if config.cuda:
#    netD = netD.cuda()
    netG = netG.cuda()
    criterionGAN = criterionGAN.cuda()
    criterionL1 = criterionL1.cuda()
    criterionMSE = criterionMSE.cuda()
    real_a = real_a.cuda()
    real_b = real_b.cuda()

real_a = Variable(real_a)
real_b = Variable(real_b)

all_features_a = []
all_features_b = []

scaler = GradScaler()
criterion = nn.MSELoss()
optimizer = optim.Adam(netG.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

print(f'Number of iterations: {len(training_data_loader)}')

for epoch in range(config.epochs):
    epoch_loss = 0.0
    for iteration, batch in enumerate(training_data_loader, 1):
        if batch is not None:
            #print(f'batch[0] shape: {batch[0].shape}')
            if iteration > 100:  # Limit the number of batches for t-SNE to avoid memory issues
                break
            real_a_cpu, real_b_cpu = batch[0], batch[1]
            real_a.resize_(real_a_cpu.size()).copy_(real_a_cpu)
            real_b.resize_(real_b_cpu.size()).copy_(real_b_cpu)
            #outputGA = netG(real_a)
            #print(f'Real a CPU shape: {real_a_cpu.shape}')
            # Zero the parameter gradients
            optimizer.zero_grad()

            with autocast():
                coarse_s, fine_s, coarse_t, fine_t = netG(real_a, real_b)
                #print(f'Coarse s shape: {coarse_s.shape} Fine s shape: {fine_s.shape} Coarse t shape: {coarse_t.shape} Fine t shape: {fine_t.shape}')
                # Calculate loss
                loss = criterion(fine_s, real_b)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Update epoch loss
                epoch_loss += loss.item()

            # # Print iteration info
            # if iteration % 10 == 0:
            #     print(
            #         f"Epoch [{epoch + 1}/{config.epochs}], Iteration [{iteration + 1}/{len(training_data_loader)}], Loss: {loss.item():.4f}")

            # Print epoch info
        print(f"Epoch [{epoch + 1}/{config.epochs}], Average Loss: {epoch_loss / len(training_data_loader):.4f}")

    example_image = fine_s[0]  # Shape: [1, 256, 256]
    #print(f'Example image shape: {example_image.shape}')
    # Step 2: Squeeze the channel dimension
    example_image = example_image.squeeze(0)  # Shape: [256, 256]

    # Step 3: Convert to NumPy array (optional)
    example_image_np = example_image.detach().cpu().numpy()

    # Step 4: Normalize the image data to range [0, 255] for display
    example_image_np = (example_image_np - example_image_np.min()) / (
                example_image_np.max() - example_image_np.min())
    example_image_np = (example_image_np * 255).astype(np.uint8)
    #print(example_image_np)

    # Step 5: Convert to a PIL image
    bw_image = Image.fromarray(example_image_np, mode='L')

    # Save the image (optional)
    bw_image.save("black_and_white_image.png")

    # Display the image
    plt.imshow(bw_image, cmap='gray')
    plt.axis('off')  # Hide axes
    plt.show()






# # Concatenate all features
# print(f'All features classified from a shape: {coarse_a.shape}')
# print(f'All features classified from b shape: {coarse_b.shape}')


# # Apply t-SNE
# tsne = TSNE(n_components=2, perplexity=1, n_iter=250)
# coarse_a_reshaped = pooled_ca.view(pooled_ca.shape[1], -1).detach().cpu().numpy()
# coarse_b_reshaped = pooled_cb.view(pooled_cb.shape[1], -1).detach().cpu().numpy()
# print(f'Coarse a reshaped shape: {coarse_a_reshaped.shape} Coarse b reshaped shape: {coarse_b_reshaped.shape}')
# tsne_results_a = tsne.fit_transform(coarse_a_reshaped)
# tsne_results_b = tsne.fit_transform(coarse_b_reshaped)
# print(f't-SNE results a shape: {tsne_results_a.shape} t-SNE results b shape: {tsne_results_b.shape}')
# # Plot t-SNE
# plt.figure(figsize=(10, 8))
# plt.scatter(tsne_results_a, tsne_results_b, s=5, cmap='tab10')
# plt.title('t-SNE of Extracted Features')
# plt.xlabel('t-SNE Dimension 1')
# plt.ylabel('t-SNE Dimension 2')
# plt.show()
