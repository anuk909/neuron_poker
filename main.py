"""
neuron poker

Usage:
  main.py selfplay random [options]
  main.py selfplay keypress [options]
  main.py selfplay consider_equity [options]
  main.py selfplay equity_improvement --improvement_rounds=<> [options]
  main.py selfplay dqn_train [options]
  main.py selfplay dqn_play [options]
  main.py learn_table_scraping [options]

options:
  -h --help                 Show this screen.
  -r --render               render screen
  -c --use_cpp_montecarlo   use cpp implementation of equity calculator. Requires cpp compiler but is 500x faster
  -f --funds_plot           Plot funds at end of episode
  --log                     log file
  --name=<>                 Name of the saved model
  --screenloglevel=<>       log level on screen
  --episodes=<>             number of episodes to play
  --stack=<>                starting stack for each player [default: 500].
  --silent                  Run in silent mode with minimal logging

"""

import logging

import gymnasium as gym
import numpy as np
import pandas as pd
from docopt import docopt

from gym_env.env import PlayerShell
from tools.helper import get_config
from tools.helper import init_logger


# pylint: disable=import-outside-toplevel


def command_line_parser():
    """Entry function"""
    args = docopt(__doc__)
    if args["--log"]:
        logfile = args["--log"]
    else:
        print("Using default log file")
        logfile = "default"
    model_name = args["--name"] if args["--name"] else "dqn1"
    screenloglevel = (
        logging.CRITICAL  # Use CRITICAL for silent mode
        if args["--silent"]
        else (
            logging.INFO
            if not args["--screenloglevel"]
            else getattr(logging, args["--screenloglevel"].upper())
        )
    )
    _ = get_config()
    init_logger(screenlevel=screenloglevel, filename=logfile)
    
    # For DQN training, suppress noisy loggers but keep important ones
    if args.get("dqn_train"):
        # Suppress gym environment logs but keep agent logs
        logging.getLogger("gym_env.env").setLevel(logging.WARNING)
        logging.getLogger("gym_env.cycle").setLevel(logging.WARNING)
        logging.getLogger("agents").setLevel(logging.INFO)
        # Suppress gymnasium warnings
        logging.getLogger("gymnasium").setLevel(logging.ERROR)
        
    print(f"Screenloglevel: {screenloglevel}")
    log = logging.getLogger("")
    log.info("Initializing program")

    if args["selfplay"]:
        num_episodes = 1 if not args["--episodes"] else int(args["--episodes"])
        runner = SelfPlay(
            render=args["--render"],
            num_episodes=num_episodes,
            use_cpp_montecarlo=args["--use_cpp_montecarlo"],
            funds_plot=args["--funds_plot"],
            stack=int(args["--stack"]),
        )

        if args["random"]:
            runner.random_agents()

        elif args["keypress"]:
            runner.key_press_agents()

        elif args["consider_equity"]:
            runner.equity_vs_random()

        elif args["equity_improvement"]:
            improvement_rounds = int(args["--improvement_rounds"])
            runner.equity_self_improvement(improvement_rounds)

        elif args["dqn_train"]:
            # For training, use WARNING level to reduce environment noise but keep important info
            if not args["--screenloglevel"]:
                screenloglevel = logging.WARNING
            runner.dqn_train(model_name)

        elif args["dqn_play"]:
            runner.dqn_play(model_name)

    else:
        raise RuntimeError("Argument not yet implemented")


class SelfPlay:
    """Orchestration of playing against itself"""

    def __init__(self, render, num_episodes, use_cpp_montecarlo, funds_plot, stack=500):
        """Initialize"""
        self.winner_in_episodes = []
        self.use_cpp_montecarlo = use_cpp_montecarlo
        self.funds_plot = funds_plot
        self.render = render
        self.env = None
        self.num_episodes = num_episodes
        self.stack = stack
        self.log = logging.getLogger(__name__)

    def random_agents(self):
        """Create an environment with 6 random players"""
        from agents.agent_random import Player as RandomPlayer

        env_name = "neuron_poker-v0"
        num_of_plrs = 2
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        for _ in range(num_of_plrs):
            player = RandomPlayer()
            self.env.unwrapped.add_player(player)

        self.env.reset()

    def key_press_agents(self):
        """Create an environment with 6 key press agents"""
        from agents.agent_keypress import Player as KeyPressAgent

        env_name = "neuron_poker-v0"
        num_of_plrs = 2
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        for _ in range(num_of_plrs):
            player = KeyPressAgent()
            self.env.unwrapped.add_player(player)

        self.env.reset()

    def equity_vs_random(self):
        """Create 6 players, 4 of them equity based, 2 of them random"""
        from agents.agent_consider_equity import Player as EquityPlayer
        from agents.agent_random import Player as RandomPlayer

        env_name = "neuron_poker-v0"
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        self.env.unwrapped.add_player(
            EquityPlayer(name="equity/50/50", min_call_equity=0.5, min_bet_equity=-0.5)
        )
        self.env.unwrapped.add_player(
            EquityPlayer(name="equity/50/80", min_call_equity=0.8, min_bet_equity=-0.8)
        )
        self.env.unwrapped.add_player(
            EquityPlayer(name="equity/70/70", min_call_equity=0.7, min_bet_equity=-0.7)
        )
        self.env.unwrapped.add_player(
            EquityPlayer(name="equity/20/30", min_call_equity=0.2, min_bet_equity=-0.3)
        )
        self.env.unwrapped.add_player(RandomPlayer())
        self.env.unwrapped.add_player(RandomPlayer())

        for _ in range(self.num_episodes):
            self.env.reset()
            self.winner_in_episodes.append(self.env.winner_ix)

        league_table = pd.Series(self.winner_in_episodes).value_counts()
        best_player = league_table.index[0]

        print("League Table")
        print("============")
        print(league_table)
        print(f"Best Player: {best_player}")

    def equity_self_improvement(self, improvement_rounds):
        """Create 6 players, 4 of them equity based, 2 of them random"""
        from agents.agent_consider_equity import Player as EquityPlayer

        calling = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        betting = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        for improvement_round in range(improvement_rounds):
            env_name = "neuron_poker-v0"
            self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
            for i in range(6):
                self.env.unwrapped.add_player(
                    EquityPlayer(
                        name=f"Equity/{calling[i]}/{betting[i]}",
                        min_call_equity=calling[i],
                        min_bet_equity=betting[i],
                    )
                )

            for _ in range(self.num_episodes):
                self.env.reset()
                self.winner_in_episodes.append(self.env.winner_ix)

            league_table = pd.Series(self.winner_in_episodes).value_counts()
            best_player = int(league_table.index[0])
            print(league_table)
            print(f"Best Player: {best_player}")

            # self improve:
            self.log.info(f"Self improvment round {improvement_round}")
            for i in range(6):
                calling[i] = np.mean([calling[i], calling[best_player]])
                self.log.info(f"New calling for player {i} is {calling[i]}")
                betting[i] = np.mean([betting[i], betting[best_player]])
                self.log.info(f"New betting for player {i} is {betting[i]}")

    def dqn_train(self, model_name):
        """Implementation of Stable Baselines3 DQN training."""
        from agents.agent_consider_equity import Player as EquityPlayer
        from agents.agent_stable_baseline3_dqn import Player as SB3Player
        from agents.agent_random import Player as RandomPlayer

        env_name = "neuron_poker-v0"
        env = gym.make(
            env_name,
            initial_stacks=self.stack,
            funds_plot=self.funds_plot,
            render=self.render,
            use_cpp_montecarlo=self.use_cpp_montecarlo,
        )

        np.random.seed(123)
        env.unwrapped.add_player(
            EquityPlayer(name="equity/50/70", min_call_equity=0.5, min_bet_equity=0.7)
        )
        env.unwrapped.add_player(
            EquityPlayer(name="equity/20/30", min_call_equity=0.2, min_bet_equity=0.3)
        )
        env.unwrapped.add_player(RandomPlayer())
        env.unwrapped.add_player(RandomPlayer())
        env.unwrapped.add_player(RandomPlayer())
        env.unwrapped.add_player(
            PlayerShell(name="sb3-dqn", stack_size=self.stack)
        )  # shell is used for callback to stable baselines3

        env.reset(seed=123)

        sb3_player = SB3Player(name="sb3-dqn")
        sb3_player.initiate_agent(env)
        sb3_player.train(env_name=model_name)

    def dqn_play(self, model_name):
        """Create 6 players, one of them a trained DQN using Stable Baselines3"""
        from agents.agent_consider_equity import Player as EquityPlayer
        from agents.agent_stable_baseline3_dqn import Player as SB3Player
        from agents.agent_random import Player as RandomPlayer

        env_name = "neuron_poker-v0"
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        self.env.unwrapped.add_player(
            EquityPlayer(name="equity/50/50", min_call_equity=0.5, min_bet_equity=0.5)
        )
        self.env.unwrapped.add_player(
            EquityPlayer(name="equity/50/80", min_call_equity=0.8, min_bet_equity=0.8)
        )
        self.env.unwrapped.add_player(
            EquityPlayer(name="equity/70/70", min_call_equity=0.7, min_bet_equity=0.7)
        )
        self.env.unwrapped.add_player(
            EquityPlayer(name="equity/20/30", min_call_equity=0.2, min_bet_equity=0.3)
        )
        self.env.unwrapped.add_player(RandomPlayer())
        self.env.unwrapped.add_player(
            PlayerShell(name="sb3-dqn", stack_size=self.stack)
        )

        self.env.reset()

        sb3_player = SB3Player(name="sb3-dqn")
        sb3_player.env = self.env
        sb3_player.load(model_name)
        sb3_player.play(nb_episodes=self.num_episodes, render=self.render)

    def dqn_train_custom_q1(self):
        """Create 6 players, 4 of them equity based, 2 of them random"""
        from agents.agent_consider_equity import Player as EquityPlayer
        from agents.agent_custom_q1 import Player as Custom_Q1
        from agents.agent_random import Player as RandomPlayer

        env_name = "neuron_poker-v0"
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        self.env.unwrapped.add_player(
            EquityPlayer(name="equity/20/30", min_call_equity=0.2, min_bet_equity=-0.3)
        )
        self.env.unwrapped.add_player(RandomPlayer())
        self.env.unwrapped.add_player(RandomPlayer())

        # Add custom Q1 player with SB3
        custom_player = Custom_Q1(name="Deep_Q1")
        custom_player.initiate_agent(self.env)
        self.env.unwrapped.add_player(custom_player)

        for _ in range(self.num_episodes):
            self.env.reset()
            self.winner_in_episodes.append(self.env.winner_ix)

        league_table = pd.Series(self.winner_in_episodes).value_counts()
        best_player = league_table.index[0]

        print("League Table")
        print("============")
        print(league_table)
        print(f"Best Player: {best_player}")


if __name__ == "__main__":
    command_line_parser()
