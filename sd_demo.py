import torch
import torch.nn as nn
import torchvision
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (AutoencoderKL, DDIMScheduler, StableDiffusionPipeline,
                       UNet2DConditionModel)


class DM(nn.Module):
    def __init__(self):
        super().__init__()

        model_kwargs = {
            "pretrained_model_name_or_path": "stabilityai/sd-turbo",
            # "pretrained_model_name_or_path": "stabilityai/sdxl-turbo",
            # 'torch_dtype': torch.float16,
            # 'variant': "fp16",
        }

        self.scheduler = DDIMScheduler.from_pretrained(subfolder="scheduler", **model_kwargs)

        self.vae = AutoencoderKL.from_pretrained(subfolder="vae", **model_kwargs)
        self.unet = UNet2DConditionModel.from_pretrained(subfolder="unet", **model_kwargs)
        self.tokenizer = CLIPTokenizer.from_pretrained(subfolder="tokenizer", **model_kwargs)
        self.text_encoder = CLIPTextModel.from_pretrained(
            subfolder="text_encoder", **model_kwargs
        )
    
    def encode_text(self, text):
        with torch.no_grad():
            text_inputs = self.tokenizer(
                text,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = self.text_encoder(
                text_input_ids.to(self.text_encoder.device), output_hidden_states=True
            )
        return prompt_embeds[0]

    def encode_image(self, image):
        # image shape = (b, c, h, w). value in [-1, 1].
        with torch.no_grad():
            z = self.vae.encode(image).latent_dist.sample()
            z = z * self.vae.config.scaling_factor
        return z

    def decode_latent(self, latents):
        # latents: (b, c, h//8, w//8)
        latents = latents / self.vae.config.scaling_factor
        image, = self.vae.decode(latents, return_dict=False)
        image = (image + 1) / 2
        image = torch.clamp(image, 0, 1)
        return image

    def forward(self, batch):
        ''' inference '''
        # how to encode_prompt, encode_image, decode_latent...
        # can be found in diffusers.StableDiffusionPipeline
        text = batch["text"]

        # encode text
        text_embed = self.encode_text(text)

        # init latent
        z = torch.randn(1, 4, 64, 64).to(text_embed.device)

        # denoise process
        self.scheduler.set_timesteps(4)
        for t in self.scheduler.timesteps:
            pred = self.unet(z, t, text_embed).sample
            z = self.scheduler.step(pred, t, z).prev_sample
        
        # decode image
        img_pred = self.decode_latent(z)
        return img_pred
    
    def training_step(self, batch, batch_idx):
        ''' 
        Haven't tested this function yet!!!
        '''
        # input
        img = batch["image"]
        text = batch["text"]

        # encode text
        text_embed = self.encode_text(text)

        # encode image
        z = self.encode_image(img)

        # diffusion
        noise = torch.randn_like(z)
        bsz = z.shape[0]
        timesteps = torch.randn(0, self.scheduler.num_train_timesteps, (bsz,), device=z.device)
        
        zt = self.scheduler.add_noise(z, noise, timesteps) 
        pred_noise = self.unet(zt, timesteps, text_embed).sample

        # loss
        loss = nn.funcitonal.mse_loss(pred_noise, noise)
        
        # log
        self.log("train_loss", loss)

        return loss

        
def simple_pipeline_demo():
    # settings
    device = "cuda:1"
    generator = torch.Generator(device=device).manual_seed(42)

    # setup model
    model_kwargs = {
            "pretrained_model_name_or_path": "stabilityai/sd-turbo",
            # 'torch_dtype': torch.float16,
            # 'variant': "fp16",
        }
    pipe = StableDiffusionPipeline.from_pretrained(**model_kwargs)
    # print(pipe.scheduler.config)
    pipe.to(device)

    # input
    prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
    # inference
    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0, generator=generator).images[0]
    image.save("generated_image.png")
    exit()

        
if __name__ == "__main__":
    # 1. Diffusers pipeline version of inference
    simple_pipeline_demo()


    # 2. Custom version of inference
    # **(a) i use ddim scheduler, thereby having lower img quality
    # **(b) Most details can be found in diffusers.StableDiffusionPipeline
    device = "cuda:1"

    gen = DM()
    gen.eval()
    gen.to(device)

    batch = {
        "text": ["a photo of a cat"]
    }

    with torch.no_grad():
        img_pred = gen.forward(batch)
        torchvision.utils.save_image(img_pred, "test.png")    


