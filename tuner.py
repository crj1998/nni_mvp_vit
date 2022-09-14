
from nni.experiment import Experiment
from nni.experiment.config import ExperimentConfig

# Define the search space of hyper-parameters
search_space = {
    'sparsity': {'_type': 'choice', '_value': [0.75, 0.5, 0.25]},
    'regular_scale': {'_type': 'choice', '_value': [1, 2, 4, 8, 16, 30]},
    'name': {'_type': 'choice', '_value': ['.attn.qkv', '.attn.proj', '.mlp', '']},
}

# Configure the experiment
config = ExperimentConfig('local')
config.experiment_name = 'ViT Tuner'
# Configure trial code
config.trial_command = 'python trial.py'
config.trial_code_directory = '.'
# Configure search space
config.search_space = search_space
# Configure tuning algorithm
config.tuner.name = 'GridSearch'
# Configure how many trials to run and gpu usage
config.max_trial_number = 3*6*4
config.trial_gpu_number = 1
config.trial_concurrency = 6
config.training_service.use_active_gpu = False

experiment = Experiment(config)
# Run the experiment
experiment.run(8080)

# thread blocking
while input('Do you want to finish [Y/n]?') != 'Y':
    pass