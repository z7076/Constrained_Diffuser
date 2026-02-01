import diffuser.utils as utils
from ml_logger import logger
import torch
from copy import deepcopy
import numpy as np
import os
import gym
from config.locomotion_config import Config
from diffuser.utils.arrays import to_torch, to_np, to_device
from diffuser.datasets.d4rl import suppress_output
import jaynes
import matplotlib.pyplot as plt
import time
import d4rl
import safety_gymnasium as sgym
import gymnasium

def evaluate(**deps):
    from ml_logger import logger, RUN
    from config.locomotion_config import Config

    RUN._update(deps)
    Config._update(deps)

    logger.remove('*.pkl')
    logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))

    Config.device = 'cuda'

    if Config.predict_epsilon:
        prefix = f'predict_epsilon_{Config.n_diffusion_steps}_1000000.0'
    else:
        prefix = f'predict_x0_{Config.n_diffusion_steps}_1000000.0'

    loadpath = os.path.join(Config.bucket, logger.prefix, 'checkpoint')
    
    if Config.save_checkpoints:
        loadpath = os.path.join(loadpath, f'state_{self.step}.pt')
    else:
        loadpath = os.path.join(loadpath, 'state.pt')
    
    state_dict = torch.load(loadpath, map_location=Config.device)

    # Load configs
    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)

    dataset_config = utils.Config(
        Config.loader,
        savepath='dataset_config.pkl',
        env=Config.dataset,
        horizon=Config.horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        use_padding=Config.use_padding,
        max_path_length=Config.max_path_length,
        include_returns=Config.include_returns,
        returns_scale=Config.returns_scale,
    )

    render_config = utils.Config(
        Config.renderer,
        savepath='render_config.pkl',
        env=Config.dataset,
    )

    dataset = dataset_config()
    renderer = render_config()

    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    if Config.diffusion == 'models.GaussianInvDynDiffusion':
        transition_dim = observation_dim
    else:
        transition_dim = observation_dim + action_dim

    model_config = utils.Config(
        Config.model,
        savepath='model_config.pkl',
        horizon=Config.horizon,
        transition_dim=transition_dim,
        cond_dim=observation_dim,
        dim_mults=Config.dim_mults,
        dim=Config.dim,
        returns_condition=Config.returns_condition,
        device=Config.device,
    )

    diffusion_config = utils.Config(
        Config.diffusion,
        savepath='diffusion_config.pkl',
        horizon=Config.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=Config.n_diffusion_steps,
        loss_type=Config.loss_type,
        clip_denoised=Config.clip_denoised,
        predict_epsilon=Config.predict_epsilon,
        hidden_dim=Config.hidden_dim,
        ar_inv=Config.ar_inv,
        train_only_inv=Config.train_only_inv,
        ## loss weighting
        action_weight=Config.action_weight,
        loss_weights=Config.loss_weights,
        loss_discount=Config.loss_discount,
        returns_condition=Config.returns_condition,
        device=Config.device,
        condition_guidance_w=Config.condition_guidance_w,
        constrained_mode=Config.constrained_mode
    )

    trainer_config = utils.Config(
        utils.Trainer,
        savepath='trainer_config.pkl',
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        log_freq=Config.log_freq,
        label_freq=int(Config.n_train_steps // Config.n_saves),
        save_parallel=Config.save_parallel,
        bucket=Config.bucket,
        n_reference=Config.n_reference,
        train_device=Config.device,
    )

    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, renderer)
    logger.print(utils.report_parameters(model), color='green')
    trainer.step = state_dict['step']
    trainer.model.load_state_dict(state_dict['model'])
    trainer.ema_model.load_state_dict(state_dict['ema'])

    num_eval = 10
    device = Config.device

    env_list = [gym.make(Config.dataset) for _ in range(num_eval)]
    dones = [0 for _ in range(num_eval)]
    episode_rewards = [0 for _ in range(num_eval)]
    episode_costs = [0 for _ in range(num_eval)]
    episode_planned_costs = [0 for _ in range(num_eval)]
    epsiode_violations = [0 for _ in range(num_eval)]

    assert trainer.ema_model.condition_guidance_w == Config.condition_guidance_w
    returns = to_device(Config.test_ret * torch.ones(num_eval, 1), device)

    t = 0
    # print([env.reset() for env in env_list])
    obs_list = [env.reset()[None] for env in env_list]
    obs = np.concatenate(obs_list, axis=0)
    recorded_obs = [deepcopy(obs[:, None])]
    num_inv_actions = 1

    while sum(dones) <  num_eval:
        obs_norm = dataset.normalizer.normalize(obs, 'observations')
        conditions = {0: to_torch(obs_norm, device=device)}
        
        start = time.time()
        # Generate sequence of K states (K=horizon, must be >= 6)
        samples = trainer.ema_model.conditional_sample(conditions, returns=returns)
        # Ensure horizon is at least 6
        if samples.shape[1] < 2*num_inv_actions:
             raise ValueError(f"Horizon ({samples.shape[1]}) must be at least 6 to generate 3 actions.")

        state_pairs = []
        for k in range(num_inv_actions):
            s_start_idx = k
            s_end_idx = s_start_idx + 1
            s_start = samples[:, s_start_idx, :]
            s_end = samples[:, s_end_idx, :]
            pair = torch.cat([s_start, s_end], dim=-1)
            state_pairs.append(pair)

        # Stack pairs and reshape for batch inverse model inference
        # Shape becomes (num_eval * num_inv_actions, 2 * obs_dim)
        obs_comb_batch = torch.cat(state_pairs, dim=0)

        # Get actions per environment instance
        # Output shape: (num_eval * num_inv_actions, action_dim)
        actions_batch = trainer.ema_model.inv_model(obs_comb_batch)
        end = time.time()

        actions_sequence = actions_batch.view(num_eval, num_inv_actions, action_dim)

        # Convert to numpy and unnormalize
        actions_sequence_np = to_np(actions_sequence)
        actions_sequence_unnormalized = dataset.normalizer.unnormalize(
            actions_sequence_np.reshape(-1, action_dim), 'actions'
        ).reshape(num_eval, num_inv_actions, action_dim)
        samples = to_np(samples)
        planned_cost = to_np(trainer.ema_model.safe)

        if t == 0:
            
            print('planned_cost', planned_cost)
            print('time', start - end)
            normed_observations = samples[:, :, :]
            observations = dataset.normalizer.unnormalize(normed_observations, 'observations')
            # print('observations', observations[:, :, 0])
            os.makedirs(os.path.join(Config.dataset + '/images'), exist_ok=True)
            savepath = os.path.join(Config.dataset + '/images', 'sample-planned.png')
            renderer.composite(savepath, observations)
        
        
        for k in range(num_inv_actions):
            obs_list = []
            action = actions_sequence_unnormalized[:, k, :]
            for i in range(num_eval):
                this_obs, this_reward, this_done, _ = env_list[i].step(action[i])

                obs_list.append(this_obs[None])

                if this_done:
                    if dones[i] == 1:
                        pass
                    else:
                        dones[i] = 1

                        episode_costs[i] += np.max((dataset.normalizer.normalize(this_obs, 'observations')[8]-0.95, 0))
                        if dataset.normalizer.normalize(this_obs, 'observations')[8]-0.95 > 0:
                            epsiode_violations[i] += 1

                        episode_rewards[i] += this_reward
                        episode_planned_costs[i] += planned_cost[i]
                        logger.print(f"Episode ({i}): {episode_rewards[i]}", color='green')
                else:
                    if dones[i] == 1:
                        pass
                    else:
                        episode_rewards[i] += this_reward

                        episode_planned_costs[i] += planned_cost[i]
                        episode_costs[i] += np.max((dataset.normalizer.normalize(this_obs, 'observations')[8]-0.95, 0))
                        if dataset.normalizer.normalize(this_obs, 'observations')[8]-0.95 > 0:
                            epsiode_violations[i] += 1

            obs = np.concatenate(obs_list, axis=0)
            recorded_obs.append(deepcopy(obs[:, None]))
            t += 1
            print(t, episode_costs, flush=True)

    recorded_obs = np.concatenate(recorded_obs, axis=1)
    savepath = os.path.join(Config.dataset + '/images', f'sample-executed.png')
    renderer.composite(savepath, recorded_obs)
    episode_rewards = np.array(episode_rewards)
    episode_costs = np.array(episode_costs)
    episode_planned_costs = np.array(episode_planned_costs)
    epsiode_violations = np.array(epsiode_violations)
    # print(epsiode_violations)
    # print(np.mean(epsiode_violations))
    # print(np.std(epsiode_violations))
    logger.print(f"average_ep_reward: {episode_rewards}, average_ep_cost: {episode_costs}, planned_cost: {episode_planned_costs}", color='green')
    logger.print(f"average_ep_reward: {np.mean(episode_rewards)}, std_ep_reward: {np.std(episode_rewards)}, average_ep_cost: {np.mean(episode_costs)}, std_ep_cost: {np.std(episode_costs)}, planned_cost: {np.mean(episode_planned_costs)}, std_planned_cost: {np.std(episode_planned_costs)}", color='green')
    logger.log_metrics_summary({'average_ep_reward':np.mean(episode_rewards), 'std_ep_reward':np.std(episode_rewards)})
    # jaynes.report({
    #     "average_ep_reward": np.mean(episode_rewards),
    #     "std_ep_reward": np.std(episode_rewards)
    # })
