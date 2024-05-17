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

# Load the configuration
config = config_loader.load_config('configs/config.yaml')

# Define Logging
def init_log():
    logging.basicConfig(level=logging.ERROR,
        format='%(asctime)s %(message)s',
        datefmt='%Y%m%d-%H:%M:%S',
        filename=os.path.join('logs', 'log_%d_%s.log'%(config.batch_size, config.star)),
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
training_data_loader = DataLoader(dataset=train_set, num_workers=config.threads, batch_size=config.batch_size, shuffle=True)

# Building Model
logging.critical(f'Number of cuda devices found: {str(torch.cuda.device_count())}')
logging.critical('===> Building model')

if (config.modelG):
    netG = torch.load(config.modelG)
else:
    netG = define_G(config.input_nc, config.output_nc, config.ngf, 'batch', False, range(torch.cuda.device_count()))
if (config.modelD):
    netD = torch.load(config.modelD)
else:
    netD = define_D(config.input_nc + config.output_nc, config.ndf, 'batch', False, range(torch.cuda.device_count()))

# Extract features
features = netG.get_features()

# Now `features` contains the outputs of the convolutional layers
for idx, feature in enumerate(features):
    print(f"Feature {idx} shape: {feature.shape}")

criterionGAN = GANLoss()
criterionL1 = nn.L1Loss()
criterionMSE = nn.MSELoss()

# setup optimizer
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

for iteration, batch in enumerate(training_data_loader, 1):
    # forward
    real_a_cpu, real_b_cpu = batch[0], batch[1]
    # logging.critical(real_a_cpu.size())
    # logging.critical(real_b_cpu.size())
    real_a.resize_(real_a_cpu.size()).copy_(real_a_cpu)
    real_b.resize_(real_b_cpu.size()).copy_(real_b_cpu)
    features = netG(real_a)
    print(type(features))
    # Now `features` contains the outputs of the convolutional layers
    for idx, feature in enumerate(features):
        print(f"Feature {idx} shape: {feature.shape}")
