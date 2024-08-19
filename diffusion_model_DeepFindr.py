# generative deep learning ex: GANS - difficult to train but better outputs
# VAE are easy to train but outputs can be blurry
# Diffusion - high quality and diverse samples , destroy inputs by adding noise then use nn to recover it
# slower compared to other models but still new ( image input = image output )
# markov chain because steps depend on each other, latent states have same dimensions as input
# steps = 1000 < larger equals slower
# need noise scheduler, neural network and time encoding step

# fit an diffusion model on image dataset

import torch.nn.functional as F

def linear_beta_scheduler(timestep, start=0.0001, end=0.02):
  return torch.linspace(start, end, timestep)

def get_index_from_list(vals, t, x_shape):
  # returns a specific index t of a passed list of values vals while considering the batch dimension
  batch_size = t.shape[0]
  out = vals.gather(-1, t.cpu())
  return = out.reshape(batch_size, *((1,)*(len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x0, t, device="cpu"):
  # takes an image and timestep as input and returns noisy version of it
  noise = torch.randn(x0)
  sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x0.shape)
  sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x0.shape)
  # mean + variance
  return sqrt_alphas_cumprod_t.to(device)*x0.to(device) \ + sqrt_one_minus_alphas_cumprod_t.to(device)*noise.to(device)

# define beta schedule
T = 200
betas = linear_beta_schedule(timestep=T)

# pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# loading data from dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.py as plt

img_size = 64 #relatively small size but makes training faster
batch_size = 128

def load_transformed_dataset():
  data_transforms = [
    transforms.Resize(img_size, img_size)),
    transforms.RandHorizontalFlip(),
    transforms.ToTensor(), # scales data into [0,1]
    transforms.Lambda(lambda t:(t*2)-1) # scale between [-1, 1]
  ]
  data_transform = transforms.Compose(data_transforms)
  train = torchvision.datasets.StandfordCars(root=".", download=True, transform=data_transform)
  test = torchvision.datasets.StandfordCars(root=".", download=True, transform=data_transform, split='test')
  return torch.utils.data.ConcatDataset([train, test]) # merges the dataset

# reverse process of load_transformed_dataset
def show_tensor_image(image):
  reverse_transforms = transforms.Compose([
    transfroms.Lambda(lambda t: (t+1)/2),
    transforms.Lambda(lambda t: t.permute(1,2,0)), #CHW to HWC
    transforms.Lambda( lambda t:t*255.),
    transforms.Lambda(lambda t: t.mupy().astype(np.uint8)),
    transforms.ToPILImage(),
  ])

#take first image of batch
if len(image.shape) == 4:
  image = image[0, :, :, :]
plt.imshow(reverse_transform(image))

data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size = BATCH_SIZE, shuffle=True, drop_last=True)

# simulate forward diffusion
image = next(iter(dataloader))[0]

plt.figure()
plt.axis('off')
num_images = 10
stepsize = int(T/num_images)

for idx in range(0, T, stepsize):
  t = torch.Tensor([idx]).type(torch.int64)
  plt.subplot(1, num_images+1, (idx/stepsize) +1)
  image, noise = forward_diffusion_sample(image, t)
  show_tensor_image(image)

# neural network model using U-Net backward process

class Block(nn.Module):
  def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
    super().__init__()
    self.time_mlp = np.Linear(time_emb_dim, out_ch)
    if up:
      self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
      self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
    else:
      self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
      slef.transform = nn.Conv2d(out_ch, out_ch, 4,2,1)
      self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
      self.pool = nn.MaxPool2d(3, stride=2)
      self.bnorm = nn.BatchNorm2d(out_ch)
      self.relu = nn.ReLU()
 
  def forward(self, x, t, ):
    # first conv
    h = self.bnorm1(self.relu(self.conv1(x)))
    # time embedding
    time_emb = self.relu(self.time_mlp(t)))
    # extend last 2 dims
    time_emb = time_emb[(...,) + (None, )*2]
    # add time channel
    h = h + time_emb
    # second conv
    h = self.bnorm2(self.relu(self.conv2(h)))
    # up or downsample
    return self.transform(h)

class SinusoidalPositionEmbeddings(nn.module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim
  def forward(self, time):
    device = time.device
    half_dim = slef.dim // 2
    embeddings = math.log(10000) / (half_dim -1)
    embeddings = time[:, None]* embeddings[None, :]
    embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim = -1)
    return embeddings # vectors that describe a position of an index in a list

class SimpleUnet(nn.Module):
  def __init__(self):
    super().__init__()
    image_size = img_size
    image_channels = 3
    down_channels = (64, 128, 256, 512, 1024)
    up_channels = (1024, 512, 256, 128, 64)
    out_dim = 1
    
    time_emb_dim = 32
    # time embedding
    self.time_mlp = nn.Sequential(
      SinusoidalPositionEmbeddings(time_emb_dim),
      nn.Linear(time_emb_dim, time_emb_dim),
      nn.ReLU()
    )
    
    # initial projection - converts image to first dimension
    self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)
    # downsample
    self.downs = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                      time_emb_dim) \
                                for i in range(len(down_channels)-1)])
    # upsample
    self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                    time_emb_dim, up=True) \
                              for i in range(len(up_channels)-1)])
    self.output = nn.Conv2d(up_channels[-1], 3, out_dim)

def forward(self, x, timestep):
  # embedded time
  t = self.time_mlp(timestep)
  # inital conv
  x = slef.conv0(x)
  # Unet
  residual_inputs = []
  for down in self.downs:
    x = down(x, t)
    residual_inputs.append(x)
  for up in self.ups:
    residual_x = residual_inputs.pop()
    # add residual x as additional channels
    x = torch.cat((x, residual_x), dim=1)
    x = up(x, t)
  return self.output(x)

model = SimpleUnet()

# loss function (L1 = mean absolute error L2 = mean square error)
def loss(model, x0, t):
  x_noisy, noise = forward_diffusion_sample(x0, t, device)
  noise_pred = model(x_noisy, t)
  return F.l1_loss(noise, noise_pred)

@torch.no_grad()
def sample_timestep(x,t):
  # calls the model to predict the noise in the image and returns the denoised image. applies noise to this image, if we are not in the last step yet
  
  betas_t = get_index_from_list(betas, t, x.shape)
  sqrt_one-minus_alphas_cumprod_t = get_index_from_list(
    sqrt_one_minus_alphas_cumprod, t, x.shpae)
  sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
  
  # call model (current image - noise predication)
  model_mean sqrt_recip_alphas_t*( x-betas_t*model(x, t) / sqrt_one_minus_alphas_cumpord_t)
  posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

  if t == 0:
    return model_mean
  else:
    noise = torch.rand_like(x)
    return model_mean + torch.sqrt(posterior_variance_t)*noise

 def sample_plot_image():
   img_size = img_size
   img = torch.randn((1, 3, img_size, img_size), device=device)
   plt.figure()
   plt.axis('off')
   num_images = 10
   stepsize = int(T/num_images)
   
   for i in range(0,T)[::-1]:
     t = torch.full((1,), i, device=device, dtype=torch.long)
     img = sample_timestep(img, t)
     img = torch.clamp(img, -1.0, 1.0)
     if i % stepsize == 0:
       plt.subplot(1, num_images, int(i/stepsize)+1)
       show_tensor_image(img.detach().cpu())
     plt.show()

# training


                                      




         


