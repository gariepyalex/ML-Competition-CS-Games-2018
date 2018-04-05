import random

class Classifier:

    def __init__(self, label_dic):
        """
        Args:
        - label_dic :  dictionary to convert between int labels and string labels.
        """
        self.labels_dic = label_dic
        self.n_classes = len(label_dic)


    def classify(self, communication):
        return random.randint(0, self.n_classes - 1)

