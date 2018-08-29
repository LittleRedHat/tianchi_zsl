from tensorboardX import SummaryWriter

class SummaryLogger:
    def __init__(self,log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def add_summary_value(self, key, value, iteration):
        if self.writer:
            self.writer.add_scalar(key, value, iteration)
        