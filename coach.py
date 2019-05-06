from rl_coach.agents.clipped_ppo_agent import ClippedPPOAgentParameters
from rl_coach.core_types import EnvironmentSteps, EnvironmentEpisodes, TrainingSteps, RunPhase
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.filters.filter import InputFilter, NoInputFilter
from rl_coach.filters.observation import ObservationNormalizationFilter
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import SimpleSchedule, ScheduleParameters, TaskParameters
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.layers import Dense

# define the environment parameters
env_params = GymVectorEnvironment(level='simple_maze_env.envs.simple_maze_env:Maze44')

# Clipped PPO
agent_params = ClippedPPOAgentParameters()
agent_params.network_wrappers['main'].input_embedders_parameters = {'observation': InputEmbedderParameters(scheme=[])}
agent_params.network_wrappers['main'].learning_rate = 0.0003
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].activation_function = 'relu'
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].scheme = [Dense(2)]
agent_params.network_wrappers['main'].middleware_parameters.scheme = [Dense(2)]
agent_params.network_wrappers['main'].middleware_parameters.activation_function = 'relu'
agent_params.network_wrappers['main'].batch_size = 5
agent_params.network_wrappers['main'].optimizer_epsilon = 1e-3
agent_params.network_wrappers['main'].adam_optimizer_beta2 = 0.99

agent_params.pre_network_filter = InputFilter()
agent_params.pre_network_filter.add_observation_filter('observation', 'normalize_observation',
                                                        ObservationNormalizationFilter(name='normalize_observation'))



schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(int(6666))
schedule_params.steps_between_evaluation_periods = EnvironmentSteps(611)
schedule_params.evaluation_steps = EnvironmentEpisodes(0)
schedule_params.heatup_steps = EnvironmentSteps(0)

graph_manager = BasicRLGraphManager(
    agent_params=agent_params,
    env_params=env_params,
    schedule_params=schedule_params)

task_params = TaskParameters(checkpoint_save_dir='./', checkpoint_save_secs=3, use_cpu=True)
graph_manager.task_parameters = task_params

print(graph_manager.task_parameters.checkpoint_save_secs, 'gm.task_parameters.checkpoint_save_secs')


imp = graph_manager.improve()
gm = graph_manager
gm.phase = RunPhase.TEST
e0 = gm.environments[0]



def action_for_obs(gm, env, obs):
    set_env_state(env, obs)
    gm.act(EnvironmentSteps(1))
    print('new state', env.env.current_state)

def set_env_state(env, new_state):
    env.states = {'observation': new_state}
    env.env.current_state = new_state

import ipdb; ipdb.set_trace()

end = 0