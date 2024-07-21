class CurriculumLearning:
    def __init__(self, config):
        self.current_difficulty = config['curriculum_learning']['initial_difficulty']
        self.max_difficulty = config['curriculum_learning'].get('max_difficulty', 10)
        self.difficulty_increment = config['curriculum_learning'].get('difficulty_increment', 0.5)
        self.reward_threshold = config['curriculum_learning'].get('reward_threshold', 50)

    def update_difficulty(self, average_reward):
        if average_reward > self.reward_threshold:
            self.current_difficulty = min(self.current_difficulty + self.difficulty_increment, self.max_difficulty)
            self.reward_threshold *= 1.1  # Increase the threshold for the next level

    def get_action_scale(self):
        return 1 + (self.current_difficulty * 1.0)  # Scales from 1 to 11 as difficulty increases