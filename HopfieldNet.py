import numpy as np

class HopfieldNetwork:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.num_neurons = np.prod(input_shape)
        self.weights = np.zeros((self.num_neurons, self.num_neurons))

    def train(self, patterns):
        """
        パターンをネットワークに記憶させる
        
        Parameters:
        patterns (list of np.array): 記憶させるパターンのリスト（2Dのバイナリ配列）
        """
        for pattern in patterns:
            flat_pattern = pattern.flatten()
            self.weights += np.outer(flat_pattern, flat_pattern)
        
        # Ensure no self-connections
        np.fill_diagonal(self.weights, 0)

    def sign(self, x):
        """
        入力が0未満の場合は-1、それ以外の場合は1を返す
        
        Parameters:
        x (float): 入力
        
        Returns:
        int: 入力が0未満の場合は-1、それ以外の場合は1
        """
        return np.where(x >= 0, 1, -1)

    def run(self, initial_state, max_steps=100):
        """
        ネットワークを初期状態から実行し、最終的な安定状態を返す

        Parameters:
        initial_state (np.array): ネットワークの初期状態（2Dのバイナリ配列）
        max_steps (int): ネットワークを実行する最大ステップ数
        
        Returns:
        np.array: 最終的な安定状態（2Dのバイナリ配列）
        """
        state = initial_state.flatten()
        for _ in range(max_steps):
            prev_state = np.copy(state)
            for i in range(self.num_neurons):
                state[i] = self.sign(np.dot(self.weights[i], state))
            if np.array_equal(state, prev_state):
                break
        return state.reshape(self.input_shape)
