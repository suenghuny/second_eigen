from ic3net_envs.predator_prey_env import PredatorPreyEnv
import argparse
import sys
import numpy as np
import signal


import pygame
import sys

# Pygame 초기화


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):

        return self.action_space.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Example GCCNet environment random agent')
    #parser.add_argument('--nagents', type=int, default=15, help="Number of agents")
    parser.add_argument('--display', action="store_true", default=True, help="Use to display environment")
    parser.add_argument('--nfriendly', type=int, default=4, help="Number of agents")
    parser.add_argument('--nenemies', type=int, default=4, help="Total number of preys in play")
    parser.add_argument('--dim', type=int, default=15, help="Dimension of box")
    parser.add_argument('--vision', type=int, default=3, help="Vision of predator")
    parser.add_argument('--moving_prey', action="store_true", default=True, help="Whether prey is fixed or moving")
    parser.add_argument('--no_stay', action="store_true", default=False, help="Whether predators have an action to stay in place")
    parser.add_argument('--mode', default='mixed', type=str, help='cooperative|competitive|mixed (default: mixed)')
    parser.add_argument('--enemy_comm', action="store_true", default=False, help="Whether prey can communicate.")
    env = PredatorPreyEnv()
    env.init_args(parser)
    args = parser.parse_args()
    screen_width, screen_height = args.dim*10, args.dim*10
    visualize = True
    if visualize == True:
        pygame.init()
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption('Predator and Prey Visualization')
        red = (255, 0, 0)
        green = (0, 255, 0)
    def signal_handler(signal, frame):
        print('You pressed Ctrl+C! Exiting gracefully.')
        if args.display:
            env.exit_render()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    env.multi_agent_init(args)
    agent = RandomAgent(env.action_space)
    episodes = 0

    while episodes < 50:
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            if visualize == True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = False
                screen.fill((0, 0, 0))

            [_ for _ in range(100000)]
            actions = []
            for _ in range(args.nfriendly):
                action = agent.act()
                actions.append(action)
            obs, reward, done, info, visualize_positional_info, positional_info, observation_matrix = env.step(actions)
            episode_reward += np.sum(reward)
            if visualize == True:
                for position_x, position_y, cat in visualize_positional_info:
                    color = green if cat == 1 else red
                    pygame.draw.circle(screen, color, (position_x, position_y), 10)
                pygame.display.flip()
        episodes += 1
        if visualize == True:
            print(episode_reward)
            pygame.quit()
            sys.exit()


    env.close()
