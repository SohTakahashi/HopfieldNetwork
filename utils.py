import numpy as np

def add_noise(pattern, noise_level):
    """
    Add random noise to a pattern.
    
    Parameters:
    pattern (np.array): The pattern to add noise to.
    noise_level (float): The proportion of neurons to flip.
    
    Returns:
    np.array: The noisy pattern.
    """
    noisy_pattern = np.copy(pattern)
    num_neurons = noisy_pattern.size
    num_noise_neurons = int(noise_level * num_neurons)
    noise_indices = np.random.choice(num_neurons, num_noise_neurons, replace=False)
    noisy_pattern.flat[noise_indices] *= -1
    return noisy_pattern