import os

class Visualizer:
    def __init__(self, save_dir):
        self.log_path = os.path.join(save_dir, 'log.txt')

    def log_print(self, log):
        print(log)
        with open(self.log_path, "a") as log_file:
            log_file.write('%s\n' % log)

