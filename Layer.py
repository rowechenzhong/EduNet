class Layer:
    def __init__(self):
        self.o_size = None

    def micro(self):
        pass

    def update_input(self, i_size):
        pass

    def get_output(self):
        return self.o_size

    def propagate(self, A):
        pass

    def backpropagate(self, dLdA):
        pass
