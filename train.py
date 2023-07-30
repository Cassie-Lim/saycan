import warnings
warnings.filterwarnings('ignore')

import collections
import datetime
import os
import random
import threading
import time

import torch
import pickle
import clip
import numpy as np
import jax
import jax.numpy as jnp
import flax
from flax.training import checkpoints
from flax.metrics import tensorboard
from model import TransporterNets, train_step, n_params
from llm import encode_text
from constants import *
def main():
    # Load data
    dataset = pickle.load(open('dataset-9999.pkl', 'rb'))  # ~10K samples.
    dataset_size = len(dataset['text'])
    #@markdown Compute CLIP features for text in the dataset.

    # Precompute CLIP features for all text in training dataset.
    text_tokens = clip.tokenize(dataset['text']).cuda()
    text_i = 0
    data_text_feats = np.zeros((0, 512), dtype=np.float32)
    while text_i < len(text_tokens):
        batch_size = min(len(text_tokens) - text_i, 512)
        text_batch = text_tokens[text_i:text_i+batch_size]
        with torch.no_grad():
            batch_feats = encode_text(text_batch).float()
        batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
        batch_feats = np.float32(batch_feats.cpu())
        data_text_feats = np.concatenate((data_text_feats, batch_feats), axis=0)
        text_i += batch_size

    #@markdown **TensorBoard:** Displays an interactive TensorBoard.
    name = datetime.datetime.now().strftime(f'%Y-%m-%d-%H:%M:%S-cliport')
    logdir = os.path.join("logs", name)
    writer = tensorboard.SummaryWriter(logdir)


    # Initialize model weights using dummy tensors.
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    init_img = jnp.ones((4, 224, 224, 5), jnp.float32)
    init_text = jnp.ones((4, 512), jnp.float32)
    init_pix = jnp.zeros((4, 2), np.int32)
    init_params = TransporterNets().init(key, init_img, init_text, init_pix)['params']
    print(f'Model parameters: {n_params(init_params):,}')
    optim = flax.optim.Adam(learning_rate=1e-4).create(init_params)
    train(optim, dataset, rng, data_text_feats, dataset_size, writer)


def train(optim, dataset, rng, data_text_feats, dataset_size, writer, batch_size=8, epochs=4000):
    # Training loop.
    for train_iter in range(1, epochs+1):
        batch_i = np.random.randint(dataset_size, size=batch_size)
        text_feat = data_text_feats[batch_i, ...]
        img = dataset['image'][batch_i, ...] / 255
        img = np.concatenate((img, np.broadcast_to(coords[None, ...], (batch_size,) + coords.shape)), axis=3)

        # Get onehot label maps.
        pick_yx = np.zeros((batch_size, 2), dtype=np.int32)
        pick_onehot = np.zeros((batch_size, 224, 224), dtype=np.float32)
        place_onehot = np.zeros((batch_size, 224, 224), dtype=np.float32)
        for i in range(len(batch_i)):
            pick_y, pick_x  = dataset['pick_yx'][batch_i[i], :]
            place_y, place_x = dataset['place_yx'][batch_i[i], :]
            pick_onehot[i, pick_y, pick_x] = 1
            place_onehot[i, place_y, place_x] = 1
            # pick_onehot[i, ...] = scipy.ndimage.gaussian_filter(pick_onehot[i, ...], sigma=3)

            # Data augmentation (random translation).
            roll_y, roll_x = np.random.randint(-112, 112, size=2)
            img[i, ...] = np.roll(img[i, ...], roll_y, axis=0)
            img[i, ...] = np.roll(img[i, ...], roll_x, axis=1)
            pick_onehot[i, ...] = np.roll(pick_onehot[i, ...], roll_y, axis=0)
            pick_onehot[i, ...] = np.roll(pick_onehot[i, ...], roll_x, axis=1)
            place_onehot[i, ...] = np.roll(place_onehot[i, ...], roll_y, axis=0)
            place_onehot[i, ...] = np.roll(place_onehot[i, ...], roll_x, axis=1)
            pick_yx[i, 0] = pick_y + roll_y
            pick_yx[i, 1] = pick_x + roll_x

        # Backpropagate.
        batch = {}
        batch['img'] = jnp.float32(img)
        batch['text'] = jnp.float32(text_feat)
        batch['pick_yx'] = jnp.int32(pick_yx)
        batch['pick_onehot'] = jnp.float32(pick_onehot)
        batch['place_onehot'] = jnp.float32(place_onehot)
        rng, batch['rng'] = jax.random.split(rng)
        optim, loss, _, _ = train_step(optim, batch)
        writer.scalar('train/loss', loss, train_iter)

        if train_iter % np.power(10, min(4, np.floor(np.log10(train_iter)))) == 0:
            print(f'Train Step: {train_iter} Loss: {loss}')

        if train_iter % 1000 == 0:
            checkpoints.save_checkpoint('.', optim, train_iter, prefix='ckpt_', keep=100000, overwrite=True)

if __name__ == '__main__':
    main()