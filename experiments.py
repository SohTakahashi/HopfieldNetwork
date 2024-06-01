# %%
import numpy as np
from HopfieldNet import HopfieldNetwork
from utils import add_noise
import seaborn as sns
import matplotlib.pyplot as plt
# %% 記憶するパターンを生成
np.random.seed(30)
patterns = []
for i in range(100):
    pattern = np.random.choice([-1, 1], size=(5, 5))
    patterns.append(pattern)

np.save('patterns.npy', patterns)    

# パターンをプロット
plt.figure(figsize=(12, 6))
for i, pattern in enumerate(patterns[:6]):
    plt.subplot(2, 3, i+1)
    sns.heatmap(pattern, cmap='binary', cbar=False, square=True, xticklabels=False, yticklabels=False)
    plt.title(f'Pattern {i+1}')
plt.show()

noise_level = 0.2
num_trials = 1000

                
# %% 実験1: パターンを1つ記憶し、ノイズを15%加えてネットワークを実行
# ネットワークを作成
network = HopfieldNetwork(input_shape=(5, 5))

# 1つだけパターンを記憶
network.train([patterns[0]])

# run the network with noise added to the pattern
mean_similarity = 0.0
mean_accuracy = 0.0

for i in range(num_trials):
    noisy_pattern = add_noise(patterns[0], noise_level=noise_level)
    result = network.run(noisy_pattern)
    similarity = np.mean(result == patterns[0])
    mean_similarity += similarity
    mean_accuracy += similarity == 1.0


mean_similarity /= num_trials
mean_accuracy /= num_trials

print(f'Mean Similarity: {mean_similarity}')
print(f'Mean Accuracy: {mean_accuracy}')


# %% 実験2: 記憶するパターンを6種類まで徐々に増やし、ノイズを15%加えてネットワークを実行
network2 = HopfieldNetwork(input_shape=(5, 5))

mean_similarities = []
mean_accuracies = []

for i in range(1, 7):
    network2.train(patterns[:i])
    mean_similarity = 0.0
    mean_accuracy = 0.0
    for j in range(num_trials):
        noisy_pattern = add_noise(patterns[0], noise_level=noise_level)
        result = network2.run(noisy_pattern)
        similarity = np.mean(result == patterns[0])
        mean_similarity += similarity
        mean_accuracy += similarity == 1.0
    mean_similarity /= num_trials
    mean_accuracy /= num_trials
    mean_similarities.append(mean_similarity)
    mean_accuracies.append(mean_accuracy)

# 実験2の結果をプロット
plt.figure(figsize=(12, 6))
plt.plot(range(1, 7), mean_similarities, label='Mean Similarity')
plt.plot(range(1, 7), mean_accuracies, label='Mean Accuracy')
plt.ylim(-0.1, 1.1)
plt.xlabel('Number of Patterns')
plt.ylabel('Similarity/Accuracy')
plt.title('Effect of Number of Patterns on Similarity and Accuracy')
plt.legend()
plt.savefig('figures/experiment2.png')

# %% 実験3: パターンを2種類記憶し、ノイズを0%から100%まで変化させてネットワークを実行
network3 = HopfieldNetwork(input_shape=(5, 5))
network3.train(patterns[:2])

mean_similarities = []
mean_accuracies = []

for noise_level in np.linspace(0, 1, 11):
    mean_similarity = 0.0
    mean_accuracy = 0.0
    for j in range(num_trials):
        noisy_pattern = add_noise(patterns[0], noise_level=noise_level)
        result = network3.run(noisy_pattern)
        similarity = np.mean(result == patterns[0])
        mean_similarity += similarity
        mean_accuracy += similarity == 1.0
    mean_similarity /= num_trials
    mean_accuracy /= num_trials
    mean_similarities.append(mean_similarity)
    mean_accuracies.append(mean_accuracy)

# 実験3の結果をプロット
plt.figure(figsize=(12, 6))
plt.plot(np.linspace(0, 1, 11), mean_similarities, label='Mean Similarity')
plt.plot(np.linspace(0, 1, 11), mean_accuracies, label='Mean Accuracy')
plt.ylim(-0.1, 1.1)
plt.xlabel('Noise Level')
plt.ylabel('Similarity/Accuracy')
plt.title('Effect of Noise Level on Similarity and Accuracy')
plt.legend()
plt.savefig('figures/experiment3.png')

# %% 実験4: パターンを4種類記憶し、ノイズを0%から100%まで変化させてネットワークを実行
network4 = HopfieldNetwork(input_shape=(5, 5))
network4.train(patterns[:4])

mean_similarities = []
mean_accuracies = []

for noise_level in np.linspace(0, 1, 11):
    mean_similarity = 0.0
    mean_accuracy = 0.0
    for j in range(num_trials):
        noisy_pattern = add_noise(patterns[0], noise_level=noise_level)
        result = network4.run(noisy_pattern)
        similarity = np.mean(result == patterns[0])
        mean_similarity += similarity
        mean_accuracy += similarity == 1.0
    mean_similarity /= num_trials
    mean_accuracy /= num_trials
    mean_similarities.append(mean_similarity)
    mean_accuracies.append(mean_accuracy)

# 実験4の結果をプロット
plt.figure(figsize=(12, 6))
plt.plot(np.linspace(0, 1, 11), mean_similarities, label='Mean Similarity')
plt.plot(np.linspace(0, 1, 11), mean_accuracies, label='Mean Accuracy')
plt.ylim(-0.1, 1.1)
plt.xlabel('Noise Level')
plt.ylabel('Similarity/Accuracy')
plt.title('Effect of Noise Level on Similarity and Accuracy')
plt.legend()
plt.savefig('figures/experiment4.png')


# %% 実験5: ノイズは15%で固定し、パターン数を変化させてネットワークを実行
network5 = HopfieldNetwork(input_shape=(5, 5))

mean_similarities = []
mean_accuracies = []

for i in range(1, 101):
    network5.train(patterns[:i])
    mean_similarity = 0.0
    mean_accuracy = 0.0
    for j in range(num_trials):
        noisy_pattern = add_noise(patterns[0], noise_level=0.15)
        result = network5.run(noisy_pattern)
        similarity = np.mean(result == patterns[0])
        mean_similarity += similarity
        mean_accuracy += similarity == 1.0
    mean_similarity /= num_trials
    mean_accuracy /= num_trials
    mean_similarities.append(mean_similarity)
    mean_accuracies.append(mean_accuracy)

# 実験5の結果をプロット
plt.figure(figsize=(12, 6))
plt.plot(range(1, 101), mean_similarities, label='Mean Similarity')
plt.plot(range(1, 101), mean_accuracies, label='Mean Accuracy')
plt.ylim(-0.1, 1.1)
plt.xlabel('Number of Patterns')
plt.ylabel('Similarity/Accuracy')
plt.title('Effect of Number of Patterns on Similarity and Accuracy')
plt.legend()
plt.savefig('figures/experiment5.png')

# %%
