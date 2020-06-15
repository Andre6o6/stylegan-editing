import torch
import torch.nn as nn
import torch.nn.functional as F

from interfacegan.models.stylegan_generator import StyleGANGenerator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dlatent_path = "latents/"
latent_predictor_path = ""
image_path = ""
predict_initial_approximation = False

# StyleGAN is converted from official tensorflow model to pytorch
converted_model = StyleGANGenerator("stylegan_ffhq")

# Initialize latent optimization pipeline (generator > vgg)
synthesizer = converted_model.model.synthesis
latent_optimizer = LatentOptimizer(synthesizer, vgg_layer=9)

# FIXME do I need requires_grad_?
for param in latent_optimizer.parameters():
    param.requires_grad_(False)

# Load target image
reference_image = load_image(image_path)
reference_image = torch.from_numpy(reference_image).unsqueeze(0).to(device)

# Get image features and image itself to use as a variable in a computational graph
reference_features = latent_optimizer.feature_extractor(reference_image).detach()
reference_image = reference_image.detach()

# Predict initial latent vector or randomly sample it
if predict_initial_approximation:
    image_to_latent = InitialLatentPredictor().to(device)
    image_to_latent.load_state_dict(torch.load(latent_predictor_path))
    image_to_latent.eval()

    with torch.no_grad():
        initial_latents = image_to_latent(reference_image)
    initial_latents = initial_latents.to(device).requires_grad_(True)
else:
    initial_latents = torch.zeros((1,18,512)).to(device).requires_grad_(True)

# Loss and optimazer to use in latent optimization
criterion = LatentLoss()
optimizer = torch.optim.Adam([initial_latents], lr=0.025)

# Latent optimization
n_iters = 100
progress_bar = tqdm(range(n_iters))
for step in progress_bar:
    optimizer.zero_grad()

    generated_image_features, _ = latent_optimizer(initial_latents)
    
    loss = criterion(generated_image_features, reference_features)
    loss.backward()
    optimizer.step()
    progress_bar.set_description("{}/{}: Loss = {}".format(step+1, n_iters, loss.item()))

# Save optimized latent vectors
optimized_dlatents = initial_latents.detach().cpu().numpy()
np.save(dlatent_path, optimized_dlatents)

# TODO save image
# TODO save optimiaztion process as a video