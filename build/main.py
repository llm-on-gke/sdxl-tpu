import io
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import uvicorn
#from diffusers import StableDiffusionPipeline

import time

import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import replicate
from jax import pmap
from jax.experimental.compilation_cache import compilation_cache as cc
from maxdiffusion import FlaxStableDiffusionXLPipeline

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def info():
  return Response('Welcome to Max Diffusion', status_code=200)
# Let's cache the model compilation, so that it doesn't take as long the next time around.

# Load the Stable Diffusion model
cc.initialize_cache("~/jax_cache")

NUM_DEVICES = jax.device_count()

# 1. Let's start by downloading the model and loading it into our pipeline class
# Adhering to JAX's functional approach, the model's parameters are returned seperatetely and
# will have to be passed to the pipeline during inference
pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", revision="refs/pr/95", split_head_dim=True
)

# 2. We cast all parameters to bfloat16 EXCEPT the scheduler which we leave in
# float32 to keep maximal precision
scheduler_state = params.pop("scheduler")
params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
params["scheduler"] = scheduler_state

# 3. Next, we define the different inputs to the pipeline
default_prompt = "a colorful photo of a castle in the middle of a forest with trees and bushes, by Ismail Inceoglu, shadows, high contrast, dynamic shading, hdr, detailed vegetation, digital painting, digital drawing, detailed painting, a detailed digital painting, gothic art, featured on deviantart"
default_neg_prompt = "fog, grainy, purple"
default_seed = 33
default_guidance_scale = 5.0
default_num_steps = 40
width = 1024
height = 1024


# 4. In order to be able to compile the pipeline
# all inputs have to be tensors or strings
# Let's tokenize the prompt and negative prompt
def tokenize_prompt(prompt, neg_prompt):
    prompt_ids = pipeline.prepare_inputs(prompt)
    neg_prompt_ids = pipeline.prepare_inputs(neg_prompt)
    return prompt_ids, neg_prompt_ids


# 5. To make full use of JAX's parallelization capabilities
# the parameters and input tensors are duplicated across devices
# To make sure every device generates a different image, we create
# different seeds for each image. The model parameters won't change
# during inference so we do not wrap them into a function
p_params = replicate(params)


def replicate_all(prompt_ids, neg_prompt_ids, seed):
    p_prompt_ids = replicate(prompt_ids)
    p_neg_prompt_ids = replicate(neg_prompt_ids)
    rng = jax.random.PRNGKey(seed)
    rng = jax.random.split(rng, NUM_DEVICES)
    return p_prompt_ids, p_neg_prompt_ids, rng


# 6. To compile the pipeline._generate function, we must pass all parameters
# to the function and tell JAX which are static arguments, that is, arguments that
# are known at compile time and won't change. In our case, it is num_inference_steps,
# height, width and return_latents.
# Once the function is compiled, these parameters are ommited from future calls and
# cannot be changed without modifying the code and recompiling.
def aot_compile(
    prompt=default_prompt,
    negative_prompt=default_neg_prompt,
    seed=default_seed,
    guidance_scale=default_guidance_scale,
    num_inference_steps=default_num_steps,
):
    prompt_ids, neg_prompt_ids = tokenize_prompt(prompt, negative_prompt)
    prompt_ids, neg_prompt_ids, rng = replicate_all(prompt_ids, neg_prompt_ids, seed)
    g = jnp.array([guidance_scale] * prompt_ids.shape[0], dtype=jnp.float32)
    g = g[:, None]

    return (
        pmap(pipeline._generate, static_broadcasted_argnums=[3, 4, 5, 9])
        .lower(
            prompt_ids,
            p_params,
            rng,
            num_inference_steps,  # num_inference_steps
            height,  # height
            width,  # width
            g,
            None,
            neg_prompt_ids,
            False,  # return_latents
        )
        .compile()
    )


start = time.time()
print("Compiling ...")
p_generate = aot_compile()
print(f"Compiled in {time.time() - start}")


# 7. Let's now put it all together in a generate function.
@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data["prompt"]
    prompt_ids, neg_prompt_ids = tokenize_prompt(prompt, default_neg_prompt)
    prompt_ids, neg_prompt_ids, rng = replicate_all(prompt_ids, neg_prompt_ids, default_seed)
    g = jnp.array([default_guidance_scale] * prompt_ids.shape[0], dtype=jnp.float32)
    g = g[:, None]
    images = p_generate(prompt_ids, p_params, rng, g, None, neg_prompt_ids)

    # convert the images to PIL
    images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
    buffer = io.BytesIO()
    images[0].save(buffer, format="PNG")

    # Return the image as a response
    return Response(content=buffer.getvalue(), media_type="image/png")

