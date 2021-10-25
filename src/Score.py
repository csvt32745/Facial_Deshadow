import numpy as np

class Losser():
    def __init__(self):
        self.loss = []

    def add(self, loss_item):
        self.loss.append(loss_item)

    def mean(self):
        return sum(self.loss) / len(self.loss)

    def clear(self):
        self.loss.clear()

class DictLosser():
    def __init__(self):
        self.loss_dict = {}
    
    def add(self, loss_item: dict):
        if not self.loss_dict:
            self.loss_dict = { k: [v] for k, v in loss_item.items() }
            return
        for k, v in loss_item.items():
            self.loss_dict[k].append(v)
    
    def mean(self):
        return { k: np.mean(v) for k, v in self.loss_dict.items() }
    
    def clear(self):
        self.loss_dict.clear()