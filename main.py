import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Kohonen SOM Implementation (Keeps Your Code)
# -----------------------------------------------------------------------------
class Neuron:
    def __init__(self, w, posx, posy):
        self.weights = w.flatten()
        self.posx = posx
        self.posy = posy
        self.y = 0.0

    def compute(self, x):
        self.y = np.linalg.norm(self.weights - x)

    def learn(self, eta, sigma, posxjetoile, posyjetoile, x):
        sigma = max(sigma, 1e-6)  # Prevent division by zero
        dist = np.sqrt((self.posx - posxjetoile) ** 2 + (self.posy - posyjetoile) ** 2)
        influence = np.exp(-dist ** 2 / (2 * sigma ** 2))
        self.weights += eta * influence * (x - self.weights)

class SOM:
    def __init__(self, inputsize, gridsize):
        self.inputsize = inputsize
        self.gridsize = gridsize
        self.map = []
        self.weightsmap = []
        self.activitymap = np.zeros(gridsize)

        for posx in range(gridsize[0]):
            mline = []
            wmline = []
            for posy in range(gridsize[1]):
                neuron = Neuron(np.random.random(self.inputsize), posx, posy)
                mline.append(neuron)
                wmline.append(neuron.weights)
            self.map.append(mline)
            self.weightsmap.append(wmline)
        self.weightsmap = np.array(self.weightsmap)

    def compute(self, x):
        for x_pos in range(self.gridsize[0]):
            for y_pos in range(self.gridsize[1]):
                self.map[x_pos][y_pos].compute(x)
                self.activitymap[x_pos, y_pos] = self.map[x_pos][y_pos].y
        return self.activitymap  # Ensure it returns a valid matrix

    def learn(self, eta, sigma, x):
        self.compute(x)
        jetoilex, jetoiley = np.unravel_index(np.argmin(self.activitymap), self.gridsize)
        for x_pos in range(self.gridsize[0]):
            for y_pos in range(self.gridsize[1]):
                self.map[x_pos][y_pos].learn(eta, sigma, jetoilex, jetoiley, x)

    def quantification(self, X):
        return np.mean([np.min(self.compute(x.flatten())) for x in X])

    def auto_organisation(self, X):
        error_count = 0
        for x in X:
            self.compute(x.flatten())
            flat = self.activitymap.flatten()
            bmu_index = np.argmin(flat)
            temp = flat.copy()
            temp[bmu_index] = np.inf
            sbmu_index = np.argmin(temp)
            bmu_coords = np.unravel_index(bmu_index, self.gridsize)
            sbmu_coords = np.unravel_index(sbmu_index, self.gridsize)
            if max(abs(bmu_coords[0] - sbmu_coords[0]), abs(bmu_coords[1] - sbmu_coords[1])) > 1:
                error_count += 1
        return error_count / len(X)

# -----------------------------------------------------------------------------
# Robotic Arm Simulation Data (θ1, θ2 → x1, x2)
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    np.random.seed(42)  # Reproducibility

    # Generate data (θ1, θ2) angles in [0, π]
    nsamples = 1200
    samples_robot = np.random.random((nsamples, 4, 1))
    samples_robot[:, 0:2, :] *= np.pi

    # Arm segment lengths
    l1, l2 = 0.7, 0.3

    # Compute (x1, x2) hand positions
    samples_robot[:, 2, :] = l1 * np.cos(samples_robot[:, 0, :]) + l2 * np.cos(samples_robot[:, 0, :] + samples_robot[:, 1, :])
    samples_robot[:, 3, :] = l1 * np.sin(samples_robot[:, 0, :]) + l2 * np.sin(samples_robot[:, 0, :] + samples_robot[:, 1, :])

    # -----------------------------------------------------------------------------
    # Train a Kohonen SOM on (θ1, θ2, x1, x2)
    # -----------------------------------------------------------------------------
    som_robotic = SOM((4, 1), (10, 10))  # 10x10 SOM grid
    ETA, SIGMA, N = 0.05, 1.4, 30000

    print("Apprentissage de la carte de Kohonen pour le bras robotique...")
    for i in range(N + 1):
        index = np.random.randint(nsamples)
        x = samples_robot[index].flatten()
        som_robotic.compute(x)
        som_robotic.learn(ETA, SIGMA, x)
        if i % 5000 == 0:
            print(f"Progression: {i/N*100:.1f}%")

    # -----------------------------------------------------------------------------
    # Extract Learned Positions for Visualization
    # -----------------------------------------------------------------------------
    neuron_positions = np.array([[neuron.weights[2], neuron.weights[3]] for row in som_robotic.map for neuron in row])
    neuron_positions = neuron_positions.reshape((10, 10, 2))  # Reshape to 10x10 grid

    # -----------------------------------------------------------------------------
    # Visualization of Original Data + Learned Neuron Positions
    # -----------------------------------------------------------------------------
    plt.figure(figsize=(6, 6))

    # Plot original training data
    plt.scatter(samples_robot[:, 2, 0], samples_robot[:, 3, 0], c='b', s=10, alpha=0.3, label="Training Data")

    # Plot neurons in black
    plt.scatter(neuron_positions[:, :, 0].flatten(), neuron_positions[:, :, 1].flatten(), c='k', marker='o', label="Neurons (Learned)")

    # Connect neurons with black lines
    for i in range(10):
        plt.plot(neuron_positions[i, :, 0], neuron_positions[i, :, 1], 'k-', linewidth=1)  # Horizontal lines
        plt.plot(neuron_positions[:, i, 0], neuron_positions[:, i, 1], 'k-', linewidth=1)  # Vertical lines

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.title("Final Learned Neuron Positions in (x1, x2) Space")
    plt.show()

    # -----------------------------------------------------------------------------
    # Evaluation of Training
    # -----------------------------------------------------------------------------
    print("Erreur de quantification vectorielle moyenne:", som_robotic.quantification(samples_robot))
    print("Erreur d'auto-organisation (robotique):", som_robotic.auto_organisation(samples_robot))
