import gym
from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy
import gym_balanceBot
import os
from time import sleep

def main():
    # create the environment
    env = gym.make("gym_balanceBot-v0")

    if os.path.isfile("trained_model/dqn_balanceBot.zip") == False:
        # Instantiate the agent
        model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)

        # Train the agent
        model.learn(total_timesteps=int(2e5))
        # Save the agent
        model.save("trained_model/dqn_balanceBot")
        del model  # delete trained model to demonstrate loading

        # Load the trained agent
        model = DQN.load("trained_model/dqn_balanceBot")

        # Evaluate the agent
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    else:
        # Load the trained agent
        model = DQN.load("trained_model/dqn_balanceBot")

    # Enjoy trained agent
    obs = env.reset()
    for i in range(3000):
        action, states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        sleep(1./240.)

    env.close()

if __name__ == '__main__':
    main()
