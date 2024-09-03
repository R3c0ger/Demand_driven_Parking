class PathLoader:
    def __init__(self, env_type):
        self.env_type = env_type

    def load_path(self):
        if self.env_type == 'train':
            experiment_paths = ['./data/20240518_01', './data/20240521_01']
        else:
            experiment_paths = []
        return experiment_paths
