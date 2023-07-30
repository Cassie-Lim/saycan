import warnings
warnings.filterwarnings('ignore')
import pickle
from constants import PICK_TARGETS, PLACE_TARGETS
from model import ScriptedPolicy
from env import *
from moviepy.editor import ImageSequenceClip
import numpy as np
from helper import xyz_to_pix

env = PickPlaceEnv()

dataset = {}
dataset_size = 2  # Size of new dataset.
dataset['image'] = np.zeros((dataset_size, 224, 224, 3), dtype=np.uint8)
dataset['pick_yx'] = np.zeros((dataset_size, 2), dtype=np.int32)
dataset['place_yx'] = np.zeros((dataset_size, 2), dtype=np.int32)
dataset['text'] = []
policy = ScriptedPolicy(env)
data_idx = 0
while data_idx < dataset_size:
    np.random.seed(data_idx)
    num_pick, num_place = 3, 3

    # Select random objects for data collection.
    pick_items = list(PICK_TARGETS.keys())
    pick_items = np.random.choice(pick_items, size=num_pick, replace=False)
    place_items = list(PLACE_TARGETS.keys())
    for pick_item in pick_items:  # For simplicity: place items != pick items.
        place_items.remove(pick_item)
    place_items = np.random.choice(place_items, size=num_place, replace=False)
    config = {'pick': pick_items, 'place': place_items}

    # Initialize environment with selected objects.
    obs = env.reset(config)

    # Create text prompts.
    prompts = []
    for i in range(len(pick_items)):
        pick_item = pick_items[i]
        place_item = place_items[i]
        prompts.append(f'Pick the {pick_item} and place it on the {place_item}.')

    # Execute 3 pick and place actions.
    for prompt in prompts:
        act = policy.step(prompt, obs)
        dataset['text'].append(prompt)
        dataset['image'][data_idx, ...] = obs['image'].copy()
        dataset['pick_yx'][data_idx, ...] = xyz_to_pix(act['pick'])
        dataset['place_yx'][data_idx, ...] = xyz_to_pix(act['place'])
        data_idx += 1
        obs, _, _, _ = env.step(act)
        debug_clip = ImageSequenceClip(env.cache_video, fps=25)
        debug_clip.write_videofile("./videos/tmp.mp4", fps=25)
        env.cache_video = []
        if data_idx >= dataset_size:
            break

pickle.dump(dataset, open(f'dataset-{dataset_size}.pkl', 'wb'))