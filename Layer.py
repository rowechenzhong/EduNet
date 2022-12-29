import numpy as np
class Layer:
    def __init__(self):
        self.o_size = None

    def micro(self) -> str:
        pass

    def update_input(self, i_size) -> None:
        pass

    def get_output(self):
        return self.o_size

    def propagate(self, A) -> np.ndarray:
        pass

    def test(self, A) -> np.ndarray:
        """
        Test, as opposed to propagate, should not be used with backpropagate. Only testing.
        :param A:
        :return:
        """
        return self.propagate(A)

    def backpropagate(self, dLdA) -> np.ndarray:
        pass
