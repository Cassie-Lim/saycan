import warnings
warnings.filterwarnings('ignore')

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

#---------------------------------prompt---------------------------------------
#@title Prompt

termination_string = "done()"

gpt3_context = """
objects = [red block, yellow block, blue block, green bowl]
# move all the blocks to the top left corner.
robot.pick_and_place(blue block, top left corner)
robot.pick_and_place(red block, top left corner)
robot.pick_and_place(yellow block, top left corner)
done()

objects = [red block, yellow block, blue block, green bowl]
# put the yellow one the green thing.
robot.pick_and_place(yellow block, green bowl)
done()

objects = [yellow block, blue block, red block]
# move the light colored block to the middle.
robot.pick_and_place(yellow block, middle)
done()

objects = [blue block, green bowl, red block, yellow bowl, green block]
# stack the blocks.
robot.pick_and_place(green block, blue block)
robot.pick_and_place(red block, green block)
done()

objects = [red block, blue block, green bowl, blue bowl, yellow block, green block]
# group the blue objects together.
robot.pick_and_place(blue block, blue bowl)
done()

objects = [green bowl, red block, green block, red bowl, yellow bowl, yellow block]
# sort all the blocks into their matching color bowls.
robot.pick_and_place(green block, green bowl)
robot.pick_and_place(red block, red bowl)
robot.pick_and_place(yellow block, yellow bowl)
done()
"""

use_environment_description = False
gpt3_context_lines = gpt3_context.split("\n")
gpt3_context_lines_keep = []
for gpt3_context_line in gpt3_context_lines:
  if "objects =" in gpt3_context_line and not use_environment_description:
    continue
  gpt3_context_lines_keep.append(gpt3_context_line)

gpt3_context = "\n".join(gpt3_context_lines_keep)
print(gpt3_context)

#--------------------------------task and config---------------------------------
#@title Task and Config
only_plan = False

raw_input = "put all the blocks in different corners."
config = {"pick":  ["red block", "yellow block", "green block", "blue block"],
          "place": ["red bowl"]}

# raw_input = "move the block to the bowl."
# config = {'pick':  ['red block'],
#           'place': ['green bowl']}

# raw_input = "put any blocks on their matched colored bowls."
# config = {'pick':  ['yellow block', 'green block', 'blue block'],
#           'place': ['yellow bowl', 'green bowl', 'blue bowl']}

# raw_input = "put all the blocks in the green bowl."
# config = {'pick':  ['yellow block', 'green block', 'red block'],
#           'place': ['yellow bowl', 'green bowl']}

# raw_input = "stack all the blocks."
# config = {'pick':  ['yellow block', 'blue block', 'red block'],
#           'place': ['blue bowl', 'red bowl']}

# raw_input = "make the highest block stack."
# config = {'pick':  ['yellow block', 'blue block', 'red block'],
#           'place': ['blue bowl', 'red bowl']}

# raw_input = "stack all the blocks."
# config = {'pick':  ['green block', 'blue block', 'red block'],
#           'place': ['yellow bowl', 'green bowl']}

# raw_input = "put the block in all the corners."
# config = {'pick':  ['red block'],
#           'place': ['red bowl', 'green bowl']}

# raw_input = "clockwise, move the block through all the corners."
# config = {'pick':  ['red block'],
#           'place': ['red bowl', 'green bowl', 'yellow bowl']}
#-----------------------------------Setup scene-----------------------------------------
#@title Setup Scene
image_path = "./2db.png"
np.random.seed(2)
if config is None:
  pick_items = list(PICK_TARGETS.keys())
  pick_items = np.random.choice(pick_items, size=np.random.randint(1, 5), replace=False)

  place_items = list(PLACE_TARGETS.keys())[:-9]
  place_items = np.random.choice(place_items, size=np.random.randint(1, 6 - len(pick_items)), replace=False)
  config = {"pick":  pick_items,
            "place": place_items}
  print(pick_items, place_items)

obs = env.reset(config)

img_top = env.get_camera_image_top()
img_top_rgb = cv2.cvtColor(img_top, cv2.COLOR_BGR2RGB)
plt.imshow(img_top)

imageio.imsave(image_path, img_top)
#-----------------------------------------runner-------------------------------------------
#@ load model
if not only_plan:
  rng = jax.random.PRNGKey(0)
  rng, key = jax.random.split(rng)
  init_img = jnp.ones((4, 224, 224, 5), jnp.float32)
  init_text = jnp.ones((4, 512), jnp.float32)
  init_pix = jnp.zeros((4, 2), np.int32)
  init_params = TransporterNets().init(key, init_img, init_text, init_pix)['params']
  print(f'Model parameters: {n_params(init_params):,}')
  optim = flax.optim.Adam(learning_rate=1e-4).create(init_params)
  ckpt_path = f'ckpt_{40000}'
  assert os.path.exists(ckpt_path)
  optim = checkpoints.restore_checkpoint(ckpt_path, optim)
#@title Runner
plot_on = False
max_tasks = 5

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
session = tf.Session(graph=tf.Graph(), config=tf.ConfigProto(gpu_options=gpu_options))
saved_model_dir = "./image_path_v2"
_ = tf.saved_model.loader.load(session, ["serve"], saved_model_dir)

options = make_options(PICK_TARGETS, PLACE_TARGETS, termination_string=termination_string)
found_objects = vild(session, image_path, category_name_string, vild_params, plot_on=False)
scene_description = build_scene_description(found_objects)
env_description = scene_description

print(scene_description)

gpt3_prompt = gpt3_context
if use_environment_description:
  gpt3_prompt += "\n" + env_description
gpt3_prompt += "\n# " + raw_input + "\n"

all_llm_scores = []
all_affordance_scores = []
all_combined_scores = []
affordance_scores = affordance_scoring(options, found_objects, block_name="box", bowl_name="circle", verbose=False)
num_tasks = 0
selected_task = ""
steps_text = []
while not selected_task == termination_string:
  num_tasks += 1
  if num_tasks > max_tasks:
    break

  llm_scores, _ = gpt3_scoring(gpt3_prompt, options, verbose=True, engine=ENGINE, print_tokens=False)
  combined_scores = {option: np.exp(llm_scores[option]) * affordance_scores[option] for option in options}
  combined_scores = normalize_scores(combined_scores)
  selected_task = max(combined_scores, key=combined_scores.get)
  steps_text.append(selected_task)
  print(num_tasks, "Selecting: ", selected_task)
  gpt3_prompt += selected_task + "\n"

  all_llm_scores.append(llm_scores)
  all_affordance_scores.append(affordance_scores)
  all_combined_scores.append(combined_scores)

if plot_on:
  for llm_scores, affordance_scores, combined_scores, step in zip(
      all_llm_scores, all_affordance_scores, all_combined_scores, steps_text):
    plot_saycan(llm_scores, affordance_scores, combined_scores, step, show_top=10)

print('**** Solution ****')
print(env_description)
print('# ' + raw_input)
for i, step in enumerate(steps_text):
  if step == '' or step == termination_string:
    break
  print('Step ' + str(i) + ': ' + step)
  nlp_step = step_to_nlp(step)

if not only_plan:
  print('Initial state:')
  plt.imshow(env.get_camera_image())

  for i, step in enumerate(steps_text):
    if step == '' or step == termination_string:
      break
    nlp_step = step_to_nlp(step)
    print('GPT-3 says next step:', nlp_step)

    obs = run_cliport(optim, env, obs, nlp_step, f"./videos/v{i}.mp4", show_state=False)

  # Show camera image after task.
  print('Final state:')
  plt.imshow(env.get_camera_image())