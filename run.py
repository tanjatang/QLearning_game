from QLearning import QLearning
from agents.LearningPolicy import LearningPolicy
from agents.AgentManager import AgentManager
from environments.EnvironmentLoader import EnvironmentLoader

print("available maps are: {}".format(EnvironmentLoader("environments/maps").available_environments()))
# available maps are:
# 'exercise'
# 'small_vs_big_reward'
# 'mini_environment'
# 'difficult_vs_easy_path'
# 'pitfall'
# 'c_shape'
# 'positive_vs_negative_reward'
# 'difficult_path'
# 'small_environment'

# maps are located in environments/maps
map = "exercise"

# False is inference mode where agent is greedy
training_mode = True

# During inference mode, this checkpoint is loaded
checkpoint_file = "checkpoints/q_values_0.txt"

# Name for checkpoint files
save_name = "q_values_"

# have a look at LearningPolicy.py for other policies
epsilon_policy = LearningPolicy.exponentially_annealed_epsilon(1/10000, 0.0)

hyperparameters = {"alpha": 0.2, "discount": 0.99}

# Please note: Numerous other settings can be adjusted in settings.py

if training_mode:
    q = QLearning(epsilon_policy=epsilon_policy,
                  map_name=map,
                  hyperparameters=hyperparameters,
                  save_name=save_name)
    while True:
        q.train()

else:
    q = QLearning(epsilon_policy=LearningPolicy.constant_epsilon(0),
                  map_name=map)

    if checkpoint_file is None:
        raise Exception("Please specify the checkpoint file path!")

    q_values = AgentManager.load_q_values(checkpoint_file)

    while True:
        q.test(q_values=q_values)