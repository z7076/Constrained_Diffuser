import json
import numpy as np
from os.path import join
import pdb
import os

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils


class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# logger = utils.Logger(args)

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer)


def quantifying_diff(sequence_com, sequence_batch):
    diff = np.sqrt(np.linalg.norm(sequence_com - sequence_batch, axis=1))
    diff = np.sum(diff)
    return diff

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def smooth(diffusion):
    steps, horizon = diffusion.shape[0], diffusion.shape[1]
    diffusion_copy = diffusion.copy()
    for i in range(steps - 20, steps, 1):
        for j in range(5, horizon, 1):
            diffusion_copy[i,j,0:2] = np.mean(diffusion[i, j-5:j, 0:2], axis=0)
    
    return diffusion_copy

#---------------------------------- main loop ----------------------------------#
safe_batch = []
score_batch = []
comp_time = []
euclidean_batch = []
diffs = []
# sequence_batch = []
import time
for i in range(1):
    print("step: ", i, "/100")

    observation = env.reset()    #array([ 0.94875744,  8.93648809, -0.01347715,  0.06358764])
    observation = np.array([ 0.94875744,  2.93648809, -0.01347715,  0.06358764])

    if args.conditional:
        print('Resetting target')
        env.set_target()

    ## set conditioning xy position to be the goal
    target = env._target
    cond = {
        diffusion.horizon - 1: np.array([*target, 0, 0]),
    }

    ## observations for rendering
    rollout = [observation.copy()]

    total_reward = 0
    euclidean_reward = 0
    for t in range(env.max_episode_steps):

        state = env.state_vector().copy()


        if t == 0:

            cond[0] = observation
            # print("cond: ", cond)
            start = time.time()
            action, samples, diffusion_paths, safe = policy(cond, batch_size=args.batch_size)
            end = time.time()
            comp_time.append(end-start)
            safe_batch.append(safe.sum().cpu())

            actions = samples.actions[0]
            sequence = samples.observations[0]
            # sequence_batch.append(sequence[:,:2])
            diffusion_paths = diffusion_paths[0]
            # sequence_ref = np.load(join(args.savepath, 'sequence.npy'))
            # diff = quantifying_diff(sequence[:,:2], sequence_ref)
            # diffs.append(diff)


            ###############################################################################################
            diffusion_sm = diffusion_paths
            renderer.render_diffusion(join(args.savepath, f'diffusion.mp4'), diffusion_sm)
            diff_step = diffusion_sm.shape[0]  
            makedirs(join(args.savepath, 'png'))
            for kk in range(diff_step):
                imgpath = join(args.savepath, f'png/{kk}.png')
                # print(diffusion_sm[kk:kk+1].shape)
                renderer.composite(imgpath, diffusion_sm[kk:kk+1], ncol=1)


        if t < len(sequence) - 1:
            next_waypoint = sequence[t+1]
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0

        
        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])


        next_observation, reward, terminal, _ = env.step(action)
        # env.render()
        total_reward += reward
        euclidean_distance = np.linalg.norm(next_waypoint[:2] - target[:2])
        euclidean_reward += np.exp(-euclidean_distance)
        score = env.get_normalized_score(total_reward)


        rollout.append(next_observation.copy())


        if terminal or t == 382:
            break

        observation = next_observation
    print(np.array(rollout).shape)
    imgpath = join(args.savepath, f'png/action.png')
    renderer.composite(imgpath, np.expand_dims(np.array(rollout),axis=0), ncol=1)
    score_batch.append(score)
    euclidean_batch.append(euclidean_reward)
    print("euclidean_reward: ", euclidean_reward)
    print("safe: ", safe_batch)

# print('diff', diffs)
# sequence_batch = np.array(diffs)
# print("diff: ", np.mean(sequence_batch))
# print("diff std: ", np.std(sequence_batch))

