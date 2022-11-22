import numpy


class Layer:
    def __init__(self):
        self.o_size = None

    def update_input(self):
        pass

    def get_output(self):
        return self.o_size

    def propagate(self):
        pass

    def backpropagate(self):
        pass
