from gym.envs.registration import register

register(
    id='gym_balanceBot-v0',
    entry_point='gym_balanceBot.envs:BalanceBotEnv',
)
