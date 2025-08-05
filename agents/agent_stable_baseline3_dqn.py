"""Player based on a trained neural network using Stable-Baselines3"""

# pylint: disable=wrong-import-order,invalid-name,import-error,missing-function-docstring
import logging
import time
import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from gym_env.enums import Action

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import configure

import torch
import torch.nn as nn

autoplay = True  # play automatically if played against stable-baselines3

# Training hyperparameters
TOTAL_TIMESTEPS = 100000
LEARNING_RATE = 1e-3
BUFFER_SIZE = 50000
LEARNING_STARTS = 1000
BATCH_SIZE = 32
TARGET_UPDATE_INTERVAL = 1000
TRAIN_FREQ = 4
GRADIENT_STEPS = 1
EXPLORATION_FRACTION = 0.1
EXPLORATION_INITIAL_EPS = 1.0
EXPLORATION_FINAL_EPS = 0.05

log = logging.getLogger(__name__)


class ProgressCallback(BaseCallback):
    """Custom callback to show training progress"""
    
    def __init__(self, total_timesteps, check_freq=1000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq
        self.last_progress = 0
        
    def _on_step(self) -> bool:
        # Calculate progress percentage
        progress = (self.num_timesteps / self.total_timesteps) * 100
        
        # Show progress every check_freq steps or at key milestones
        if (self.num_timesteps % self.check_freq == 0) or (int(progress) > self.last_progress):
            remaining_steps = self.total_timesteps - self.num_timesteps
            print(f"Progress: {progress:.1f}% | Steps: {self.num_timesteps}/{self.total_timesteps} | Remaining: {remaining_steps}")
            self.last_progress = int(progress)
            
        return True


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for the DQN network"""

    def __init__(self, observation_space: spaces.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        # Get the input dimension
        if hasattr(observation_space, "shape") and observation_space.shape:
            input_dim = observation_space.shape[0]
        else:
            input_dim = 1000  # Fallback

        # Create the network
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class ActionMaskWrapper(gym.Wrapper):
    """Wrapper to handle legal moves constraint"""

    def __init__(self, env):
        super().__init__(env)
        self.legal_moves = None

    def step(self, action):
        # Process action to ensure it's legal
        if hasattr(self, "legal_moves") and self.legal_moves is not None:
            action = self._process_action(action)

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Store legal moves for next action
        if "legal_moves" in info:
            self.legal_moves = [move.value for move in info["legal_moves"]]

        return obs, reward, terminated, truncated, info

    def _process_action(self, action):
        """Find nearest legal action"""
        if action not in self.legal_moves:
            # Try to find the nearest legal action
            for i in range(1, 6):
                # Try action + i
                if action + i in self.legal_moves:
                    return action + i
                # Try action - i
                if action - i in self.legal_moves:
                    return action - i

            # If no nearby action is legal, return the first legal action
            if self.legal_moves:
                return self.legal_moves[0]

        return action


class Player:
    """Mandatory class with the player methods"""

    def __init__(self, name="DQN_SB3", load_model=None, env=None):
        """Initialization of an agent"""
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ""
        self.temp_stack = []
        self.name = name
        self.autoplay = True

        self.model = None
        self.env = env
        self.wrapped_env = None

        if load_model:
            self.load(load_model)

    def initiate_agent(self, env):
        """Initiate a DQN agent using Stable-Baselines3"""
        self.env = env

        # Wrap environment with action mask wrapper
        self.wrapped_env = ActionMaskWrapper(env)

        # Create the DQN model
        policy_kwargs = dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=512),
        )

        self.model = DQN(
            "MlpPolicy",
            self.wrapped_env,
            learning_rate=LEARNING_RATE,
            buffer_size=BUFFER_SIZE,
            learning_starts=LEARNING_STARTS,
            batch_size=BATCH_SIZE,
            target_update_interval=TARGET_UPDATE_INTERVAL,
            train_freq=TRAIN_FREQ,
            gradient_steps=GRADIENT_STEPS,
            exploration_fraction=EXPLORATION_FRACTION,
            exploration_initial_eps=EXPLORATION_INITIAL_EPS,
            exploration_final_eps=EXPLORATION_FINAL_EPS,
            policy_kwargs=policy_kwargs,
            verbose=1,  # Enable minimal training progress
            tensorboard_log="./tensorboard_logs/",  # Enable tensorboard for monitoring
        )

    def train(self, env_name):
        """Train a model"""
        if self.model is None:
            raise ValueError("Model not initialized. Call initiate_agent first.")

        # Create timestamped directory for logs
        timestr = time.strftime("%Y%m%d-%H%M%S") + "_" + str(env_name)
        log_dir = f"./logs/{timestr}/"
        os.makedirs(log_dir, exist_ok=True)

        # Wrap environment with Monitor for basic logging
        monitored_env = Monitor(self.wrapped_env, log_dir)
        self.model.set_env(monitored_env)

        # Create progress callback
        progress_callback = ProgressCallback(TOTAL_TIMESTEPS, check_freq=2000)

        # Set up evaluation callback with minimal logging
        eval_callback = EvalCallback(
            monitored_env,
            best_model_save_path=log_dir,
            log_path=log_dir,
            eval_freq=10000,
            deterministic=True,
            render=False,
            verbose=1,  # Show evaluation results
        )

        # Combine callbacks
        from stable_baselines3.common.callbacks import CallbackList
        callback_list = CallbackList([progress_callback, eval_callback])

        # Train the model
        print(f"🚀 Starting DQN training for {TOTAL_TIMESTEPS:,} timesteps")
        print(f"📊 Model: {env_name} | Learning Rate: {LEARNING_RATE}")
        print(f"💾 Logs: {log_dir}")
        print("=" * 60)
        
        self.model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback_list,
            tb_log_name=f"DQN_{env_name}",
        )

        print("=" * 60)
        print("✅ Training completed successfully!")
        
        # Save the model
        model_path = f"dqn_{env_name}"
        self.model.save(model_path)
        print(f"💾 Model saved to {model_path}.zip")

        # Evaluate the trained model
        print("🧪 Evaluating trained model...")
        self.evaluate(nb_episodes=5)

    def load(self, env_name):
        """Load a trained model"""
        model_path = f"dqn_{env_name}"
        try:
            self.model = DQN.load(model_path)
            log.info(f"Model loaded from {model_path}")
        except FileNotFoundError:
            log.error(f"Model file {model_path}.zip not found")
            raise

    def play(self, nb_episodes=5, render=False):
        """Let the agent play"""
        if self.model is None:
            raise ValueError("Model not initialized or loaded")

        self.evaluate(nb_episodes=nb_episodes, render=render)

    def evaluate(self, nb_episodes=5, render=False):
        """Evaluate the model"""
        if self.wrapped_env is None:
            self.wrapped_env = ActionMaskWrapper(self.env)

        self.model.set_env(self.wrapped_env)

        episode_rewards = []

        for episode in range(nb_episodes):
            obs, _ = self.wrapped_env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.wrapped_env.step(action)
                episode_reward += reward
                done = terminated or truncated

                if render:
                    self.wrapped_env.render()

            episode_rewards.append(episode_reward)
            print(f"  Episode {episode + 1}/{nb_episodes}: Reward = {episode_reward:.2f}")

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        print(f"📈 Evaluation Results:")
        print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"  Episodes: {nb_episodes}")

        return mean_reward, std_reward

    def action(self, action_space, observation, info):
        """Mandatory method that calculates the move based on the observation array and the action space."""
        if self.model is None:
            # Fallback to random action if model not loaded
            log.warning("Model not loaded, using random action")
            return np.random.choice(list(action_space))

        try:
            # Get valid actions
            this_player_action_space = {
                Action.FOLD,
                Action.CHECK,
                Action.CALL,
                Action.RAISE_POT,
                Action.RAISE_HALF_POT,
                Action.RAISE_2POT,
            }
            valid_actions = this_player_action_space.intersection(set(action_space))

            if not valid_actions:
                return None

            # Predict action using the model
            action, _states = self.model.predict(observation, deterministic=True)

            # Ensure action is valid
            if action in [act.value for act in valid_actions]:
                return Action(action)
            else:
                # Fallback to first valid action
                return list(valid_actions)[0]

        except Exception as e:
            log.error(f"Error in action prediction: {e}")
            # Fallback to random valid action
            valid_actions = list(action_space)
            return np.random.choice(valid_actions) if valid_actions else None
