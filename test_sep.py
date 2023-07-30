import collections
import datetime
import os
import random
import threading
import time

import cv2  # Used by ViLD.
import clip
from easydict import EasyDict
import flax
from flax import linen as nn
from flax.training import checkpoints
from flax.metrics import tensorboard
import imageio
from heapq import nlargest
# import IPython
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
import numpy as np
import optax
import pickle
from PIL import Image
import pybullet
import pybullet_data
import tensorflow.compat.v1 as tf
import torch
from robo_gripper import *
from vild import *
from env import *
from constants import *
from model import *
from llm import *
from helper import *
ENGINE = "text-davinci-003"  # "text-ada-001"
#Download PyBullet assets.
# if not os.path.exists('ur5e/ur5e.urdf'):
#   !gdown --id 1Cc_fDSBL6QiDvNT4dpfAEbhbALSVoWcc
#   !gdown --id 1yOMEm-Zp_DL3nItG9RozPeJAmeOldekX
#   !gdown --id 1GsqNLhEl9dd4Mc3BM0dX3MibOI1FVWNM
#   !unzip ur5e.zip
#   !unzip robotiq_2f_85.zip
#   !unzip bowl.zip

# ViLD pretrained model weights.
# !gsutil cp -r gs://cloud-tpu-checkpoints/detection/projects/vild/colab/image_path_v2 ./

# %load_ext tensorboard

# Show if JAX is using GPU.
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

# #@markdown Initialize environment

if 'env' in locals():
  # Safely exit gripper threading before re-initializing environment.
  env.gripper.running = False
  while env.gripper.constraints_thread.isAlive():
    time.sleep(0.01)
env = PickPlaceEnv()

#----------------------------test render-------------------------------

#@markdown Render images.

# Define and reset environment.
config = {'pick':  ['yellow block', 'green block', 'blue block'],
          'place': ['yellow bowl', 'green bowl', 'blue bowl']}

np.random.seed(42)
obs = env.reset(config)

plt.subplot(1, 2, 1)
img = env.get_camera_image()
plt.title('Perspective side-view')
plt.imshow(img)
plt.subplot(1, 2, 2)
img = env.get_camera_image_top()
img = np.flipud(img.transpose(1, 0, 2))
plt.title('Orthographic top-view')
plt.imshow(img)
plt.show()

# Note: orthographic cameras do not exist. But we can approximate them by
# projecting a 3D point cloud from an RGB-D camera, then unprojecting that onto
# an orthographic plane. Orthographic views are useful for spatial action maps.
plt.title('Unprojected orthographic top-view')
plt.imshow(obs['image'])
plt.show()

#----------------------------test ViLD-------------------------------
# Define and reset environment.
config = {'pick':  ['yellow block', 'green block', 'blue block'],
          'place': ['yellow bowl', 'green bowl', 'blue bowl']}

np.random.seed(42)
obs = env.reset(config)
img = env.get_camera_image_top()
img = np.flipud(img.transpose(1, 0, 2))
plt.title('ViLD Input Image')
# plt.imshow(img)
# plt.show()
imageio.imwrite('tmp.jpg', img)



#@markdown Load ViLD model.

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
session = tf.Session(graph=tf.Graph(), config=tf.ConfigProto(gpu_options=gpu_options))
saved_model_dir = "./image_path_v2"
_ = tf.saved_model.loader.load(session, ["serve"], saved_model_dir)

# found_objects = vild(session, image_path, category_name_string, vild_params, plot_on=False, prompt_swaps=prompt_swaps)
found_objects = vild(session, image_path, category_name_string, vild_params, plot_on=True, prompt_swaps=prompt_swaps)
#--------------------------Test Loading Dataset--------------------------------------
#@markdown Collect demonstrations with a scripted expert, or download a pre-generated dataset.
dataset = pickle.load(open('dataset-9999.pkl', 'rb'))  # ~10K samples.
dataset_size = len(dataset['text'])


#--------------------------------Test Loading model--------------------------
#@markdown Train your own model, or load a pretrained one.
load_pretrained = True  #@param {type:"boolean"}

# Initialize model weights using dummy tensors.
rng = jax.random.PRNGKey(0)
rng, key = jax.random.split(rng)
init_img = jnp.ones((4, 224, 224, 5), jnp.float32)
init_text = jnp.ones((4, 512), jnp.float32)
init_pix = jnp.zeros((4, 2), np.int32)
init_params = TransporterNets().init(key, init_img, init_text, init_pix)['params']
print(f'Model parameters: {n_params(init_params):,}')
optim = flax.optim.Adam(learning_rate=1e-4).create(init_params)

ckpt_path = f'ckpt_{40000}'
#   if not os.path.exists(ckpt_path):
#     !gdown --id 1Nq0q1KbqHOA5O7aRSu4u7-u27EMMXqgP
optim = checkpoints.restore_checkpoint(ckpt_path, optim)
print('Loaded:', ckpt_path)

#---------------------------------Test Scoring----------------------------------------------
#@title Test
termination_string = "done()"
query = "To pick the blue block and put it on the red block, I should:\n"

options = make_options(PICK_TARGETS, PLACE_TARGETS, termination_string=termination_string)
llm_scores, _ = gpt3_scoring(query, options, verbose=True, engine=ENGINE)

affordance_scores = affordance_scoring(options, found_objects, block_name="box", bowl_name="circle", verbose=False, termination_string=termination_string)

combined_scores = {option: np.exp(llm_scores[option]) * affordance_scores[option] for option in options}
combined_scores = normalize_scores(combined_scores)
selected_task = max(combined_scores, key=combined_scores.get)
print("Selecting: ", selected_task)
