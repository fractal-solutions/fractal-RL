// ppo-gui.js

// --- CartPoleEnvironment (from ppo-test.js) ---
class CartPoleEnvironment {
    constructor() {
        this.state = [0.0, 0.0, 0.0, 0.0]; // cart position, cart velocity, pole angle, pole velocity
        this.steps = 0;
        this.maxSteps = 2000;
        this.poleAngleThreshold = 0.4190 * 2.5;//0.2095; // ~12 degrees
        this.cartPositionThreshold = 4.8 * 1.5;//2.4;
    }

    reset() {
        this.state = Array(4).fill(0).map(() => (Math.random() - 0.5) * 0.1); // Small random initial state
        this.steps = 0;
        return this.state;
    }

    step(action) {
        // Simplified physics for demonstration
        const x = this.state[0];
        const x_dot = this.state[1];
        const theta = this.state[2];
        const theta_dot = this.state[3];

        const force = (action === 1) ? 10.0 : -10.0; // Apply force left or right

        const costheta = Math.cos(theta);
        const sintheta = Math.sin(theta);

        const temp = (force + 0.05 * theta_dot * theta_dot * sintheta) / 1.0; // Simplified mass_pole + mass_cart = 1.0

        const thetaacc = (9.8 * sintheta - costheta * temp) / (0.5 * (4/3 - 0.05 * costheta * costheta)); // Simplified length_pole = 0.5, mass_pole = 0.05
        const xacc = temp - 0.05 * thetaacc * costheta;

        // Update state (Euler integration)
        this.state[0] = x + 0.02 * x_dot;
        this.state[1] = x_dot + 0.02 * xacc;
        this.state[2] = theta + 0.02 * theta_dot;
        this.state[3] = theta_dot + 0.02 * thetaacc;

        this.steps++;

        const done =
            this.state[0] < -this.cartPositionThreshold ||
            this.state[0] > this.cartPositionThreshold ||
            this.state[2] < -this.poleAngleThreshold ||
            this.state[2] > this.poleAngleThreshold ||
            this.steps >= this.maxSteps;

        const reward = done ? 0 : 1; // Reward 1 for each step pole is upright

        return { state: this.state, reward, done };
    }
}

// --- Activations (from src/utils/activations.ts) ---
/**
 * @typedef {function(number): number} ActivationFunction
 */

/**
 * @typedef {function(number): number} ActivationDerivative
 */

const activations = {
    relu: {
        func: (x) => Math.max(0, x),
        deriv: (x) => (x > 0 ? 1 : 0)
    },
    leakyRelu: {
        func: (x, alpha = 0.01) => (x > 0 ? x : x * alpha),
        deriv: (x, alpha = 0.01) => (x > 0 ? 1 : alpha)
    },
    sigmoid: {
        func: (x) => 1 / (1 + Math.exp(-x)),
        deriv: (x) => {
            const s = 1 / (1 + Math.exp(-x));
            return s * (1 - s);
        }
    },
    tanh: {
        func: (x) => Math.tanh(x),
        deriv: (x) => 1 - Math.pow(Math.tanh(x), 2)
    },
    linear: { // For output layers where no activation is desired
        func: (x) => x,
        deriv: (x) => 1
    }
};

function getActivation(name) {
    switch (name) {
        case 'relu':
            return activations.relu;
        case 'leakyRelu':
            return activations.leakyRelu;
        case 'sigmoid':
            return activations.sigmoid;
        case 'tanh':
            return activations.tanh;
        case 'linear':
            return activations.linear;
        default:
            throw new Error(`Unknown activation function: ${name}`);
    }
}

// --- NeuralNetwork (from src/ppo/network.ts) ---
class NeuralNetwork {
    constructor(layerSizes, dropoutRate = 0.0, hiddenActivationName = 'leakyRelu') {
        this.layerSizes = layerSizes;
        this.dropoutRate = dropoutRate;
        this.hiddenActivation = getActivation(hiddenActivationName);

        this.weights = [];
        this.biases = [];
        this.weightGradients = [];
        this.biasGradients = [];

        // Adam optimizer parameters
        this.mWeights = [];
        this.vWeights = [];
        this.mBiases = [];
        this.vBiases = [];
        this.t = 0;

        this.dropoutMasks = [];
        this.activations = [];
        this.zValues = [];

        for (let i = 0; i < layerSizes.length - 1; i++) {
            const inputSize = layerSizes[i];
            const outputSize = layerSizes[i + 1];

            // He initialization for weights (good for ReLU-like activations)
            const weightLayer = Array(inputSize).fill(0).map(() =>
                Array(outputSize).fill(0).map(() =>
                    (Math.random() - 0.5) * 2 * Math.sqrt(1 / inputSize)
                )
            );
            this.weights.push(weightLayer);

            const biasLayer = Array(outputSize).fill(0.1); // Small positive bias to start
            this.biases.push(biasLayer);

            // Initialize gradients and optimizer parameters with zeros
            this.weightGradients.push(Array(inputSize).fill(0).map(() => Array(outputSize).fill(0)));
            this.biasGradients.push(Array(outputSize).fill(0));
            this.mWeights.push(Array(inputSize).fill(0).map(() => Array(outputSize).fill(0)));
            this.vWeights.push(Array(inputSize).fill(0).map(() => Array(outputSize).fill(0)));
            this.mBiases.push(Array(outputSize).fill(0));
            this.vBiases.push(Array(outputSize).fill(0));
        }
    }
    // Softmax activation for output layer of actor (for probabilities)
    softmax(inputs) {
        const exp = inputs.map(Math.exp);
        const sumExp = exp.reduce((a, b) => a + b, 0);
        return exp.map(e => e / sumExp);
    }

    // Forward pass through the network
    forward(input, isOutputLayerActor = false, isTraining = true) {
        if (input.length !== this.layerSizes[0]) {
            throw new Error(`Input size mismatch. Expected ${this.layerSizes[0]}, got ${input.length}`);
        }

        this.activations = [];
        this.zValues = [];
        if (isTraining) {
            this.dropoutMasks = [];
        }

        let currentActivations = input;
        this.activations.push(currentActivations); // Store input activations

        for (let i = 0; i < this.weights.length; i++) {
            const nextZValues = Array(this.layerSizes[i + 1]).fill(0);

            for (let j = 0; j < this.layerSizes[i + 1]; j++) { // For each output node in the current layer
                let sum = this.biases[i][j];
                for (let k = 0; k < this.layerSizes[i]; k++) { // For each input node from the previous layer
                    sum += currentActivations[k] * this.weights[i][k][j];
                }
                nextZValues[j] = sum;
            }
            this.zValues.push(nextZValues); // Store pre-activation values

            // Apply activation function
            if (i < this.weights.length - 1) { // Not the output layer
                currentActivations = nextZValues.map(this.hiddenActivation.func);

                // Apply Dropout
                if (isTraining && this.dropoutRate > 0) {
                    const mask = Array(currentActivations.length).fill(false).map(() => Math.random() > this.dropoutRate);
                    this.dropoutMasks.push(mask);
                    currentActivations = currentActivations.map((val, idx) => mask[idx] ? val / (1 - this.dropoutRate) : 0);
                }
            } else { // Output layer
                if (isOutputLayerActor) {
                    currentActivations = this.softmax(nextZValues); // For actor, output probabilities
                } else {
                    currentActivations = nextZValues; // For critic, output raw value
                }
            }
            this.activations.push(currentActivations); // Store layer activations
        }
        return currentActivations;
    }

    // Backpropagation
    // `outputGradient` is the gradient of the loss with respect to the network's output.
    backward(outputGradient, isOutputLayerActor = false) {
        // Gradients should be zeroed before calling backward.

        let currentDelta = outputGradient;

        // Iterate backwards through the layers
        for (let i = this.weights.length - 1; i >= 0; i--) {
            const currentLayerActivations = this.activations[i + 1]; // Activations of the current layer
            const prevLayerActivations = this.activations[i];     // Activations of the previous layer (input to current layer)
            const currentZValues = this.zValues[i];

            // Apply derivative of activation function (if not output layer or if not softmax for actor)
            if (i < this.weights.length - 1) { // Hidden layer
                // Backward pass for Dropout
                if (this.dropoutRate > 0 && this.dropoutMasks[i]) {
                    currentDelta = currentDelta.map((d, idx) => this.dropoutMasks[i][idx] ? d / (1 - this.dropoutRate) : 0);
                }

                currentDelta = currentDelta.map((d, idx) => d * this.hiddenActivation.deriv(currentZValues[idx]));
            } else if (isOutputLayerActor) {
                // For softmax, the gradient calculation is often combined with the loss function (e.g., cross-entropy)
                // Here, we assume `outputGradient` already incorporates the derivative of softmax if applicable.
                // If `outputGradient` is from a cross-entropy loss after softmax, it's typically (predicted_prob - target_prob).
                // For a general case, this is complex. We'll treat it as a linear layer for gradient propagation for simplicity.
                // A more accurate implementation would involve the Jacobian of softmax.
            }

            // Calculate gradients for biases
            for (let j = 0; j < this.biasGradients[i].length; j++) {
                this.biasGradients[i][j] += currentDelta[j];
            }

            // Calculate gradients for weights
            for (let j = 0; j < this.weightGradients[i].length; j++) { // Input nodes to current layer
                for (let k = 0; k < this.weightGradients[i][j].length; k++) { // Output nodes of current layer
                    this.weightGradients[i][j][k] += prevLayerActivations[j] * currentDelta[k];
                }
            }

            // Propagate delta backwards to the previous layer
            const nextDelta = Array(this.layerSizes[i]).fill(0);
            for (let j = 0; j < this.layerSizes[i]; j++) { // For each node in the previous layer
                for (let k = 0; k < this.layerSizes[i + 1]; k++) { // For each node in the current layer
                    nextDelta[j] += currentDelta[k] * this.weights[i][j][k];
                }
            }
            currentDelta = nextDelta;
        }
    }

    // Apply gradients using the Adam optimizer with L2 regularization and Gradient Clipping
    applyGradients(learningRate, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, l2RegularizationRate = 0.001, clipNorm = 1.0) {
        this.t++;
        const lr_t = learningRate * Math.sqrt(1 - Math.pow(beta2, this.t)) / (1 - Math.pow(beta1, this.t));

        // Calculate total gradient norm for clipping
        let totalGradNorm = 0;
        for (let i = 0; i < this.weights.length; i++) {
            for (let j = 0; j < this.weights[i].length; j++) {
                for (let k = 0; k < this.weights[i][j].length; k++) {
                    totalGradNorm += Math.pow(this.weightGradients[i][j][k], 2);
                }
            }
            for (let j = 0; j < this.biases[i].length; j++) {
                totalGradNorm += Math.pow(this.biasGradients[i][j], 2);
            }
        }
        totalGradNorm = Math.sqrt(totalGradNorm);

        const clipFactor = totalGradNorm > clipNorm ? clipNorm / totalGradNorm : 1.0;

        for (let i = 0; i < this.weights.length; i++) {
            for (let j = 0; j < this.weights[i].length; j++) {
                for (let k = 0; k < this.weights[i][j].length; k++) {
                    // Apply gradient clipping
                    const clippedWeightGrad = this.weightGradients[i][j][k] * clipFactor;

                    // L2 Regularization (Weight Decay)
                    const weightDecayGrad = l2RegularizationRate * this.weights[i][j][k];

                    // Update biased first moment estimate
                    this.mWeights[i][j][k] = beta1 * this.mWeights[i][j][k] + (1 - beta1) * (clippedWeightGrad + weightDecayGrad);
                    // Update biased second moment estimate
                    this.vWeights[i][j][k] = beta2 * this.vWeights[i][j][k] + (1 - beta2) * Math.pow((clippedWeightGrad + weightDecayGrad), 2);

                    // Correct bias
                    const mHat = this.mWeights[i][j][k];
                    const vHat = this.vWeights[i][j][k];

                    // Update weights
                    this.weights[i][j][k] -= lr_t * mHat / (Math.sqrt(vHat) + epsilon);
                }
            }
            for (let j = 0; j < this.biases[i].length; j++) {
                // Apply gradient clipping
                const clippedBiasGrad = this.biasGradients[i][j] * clipFactor;

                // Update biased first moment estimate
                this.mBiases[i][j] = beta1 * this.mBiases[i][j] + (1 - beta1) * clippedBiasGrad;
                // Update biased second moment estimate
                this.vBiases[i][j] = beta2 * this.vBiases[i][j] + (1 - beta2) * Math.pow(clippedBiasGrad, 2);

                // Correct bias
                const mHat = this.mBiases[i][j];
                const vHat = this.vBiases[i][j];

                // Update biases
                this.biases[i][j] -= lr_t * mHat / (Math.sqrt(vHat) + epsilon);
            }
        }
    }

    // Reset gradients to zero
    zeroGrad() {
        for (let i = 0; i < this.weightGradients.length; i++) {
            for (let j = 0; j < this.weightGradients[i].length; j++) {
                this.weightGradients[i][j].fill(0);
                this.mWeights[i][j].fill(0);
                this.vWeights[i][j].fill(0);
            }
            this.biasGradients[i].fill(0);
            this.mBiases[i].fill(0);
            this.vBiases[i].fill(0);
        }
        this.t = 0;
        this.dropoutMasks = []; // Clear dropout masks
    }

    // Method to copy parameters from another NeuralNetwork instance
    copyParametersFrom(otherNetwork) {
        if (this.layerSizes.toString() !== otherNetwork.layerSizes.toString()) {
            throw new Error("Cannot copy parameters: network architectures do not match.");
        }
        // Deep copy weights
        this.weights = otherNetwork.weights.map(layer => layer.map(row => [...row]));
        // Deep copy biases
        this.biases = otherNetwork.biases.map(layer => [...layer]);
    }
}

class ActorNetwork extends NeuralNetwork {
    constructor(inputSize, outputSize, hiddenLayers = [64, 64], dropoutRate = 0.0, hiddenActivationName = 'leakyRelu') {
        super([inputSize, ...hiddenLayers, outputSize], dropoutRate, hiddenActivationName);
    }

    forward(input, isOutputLayerActor = true, isTraining = true) {
        return super.forward(input, isOutputLayerActor, isTraining);
    }

    backward(outputGradient) {
        super.backward(outputGradient, true);
    }
}

class CriticNetwork extends NeuralNetwork {
    constructor(inputSize, hiddenLayers = [64, 64], dropoutRate = 0.0, hiddenActivationName = 'leakyRelu') {
        super([inputSize, ...hiddenLayers, 1], dropoutRate, hiddenActivationName); // Critic network outputs a single value
    }

    forward(input, isOutputLayerActor = false, isTraining = true) {
        return super.forward(input, isOutputLayerActor, isTraining);
    }

    backward(outputGradient) {
        super.backward(outputGradient, false);
    }
}

// --- PPOAgent (from src/ppo/agent.ts) ---
/**
 * @typedef {object} PPOAgentConfig
 * @property {number} stateDim
 * @property {number} actionDim
 * @property {number} learningRateActor
 * @property {number} learningRateCritic
 * @property {number} gamma
 * @property {number} lambda
 * @property {number} epsilon
 * @property {number} ppoEpochs
 * @property {number} miniBatchSize
 * @property {number} dropoutRate
 * @property {number} l2RegularizationRate
 * @property {number} clipNorm
 * @property {string} [hiddenActivation] - Optional: Name of the hidden activation function (e.g., 'relu', 'leakyRelu')
 */

// Represents a single experience tuple
/**
 * @typedef {object} Experience
 * @property {number[]} state
 * @property {number} action
 * @property {number} reward
 * @property {number[]} nextState
 * @property {boolean} done
 * @property {number} logProb - Log probability of the action taken by the old policy
 * @property {number} value - Value estimate of the state by the old critic
 */

class PPOAgent {
    constructor(config) {
        this.config = config;
        this.experiences = [];

        const hiddenLayers = [64, 64]; // A common default architecture

        this.actor = new ActorNetwork(
            config.stateDim,
            config.actionDim,
            hiddenLayers,
            config.dropoutRate,
            config.hiddenActivation
        );

        this.critic = new CriticNetwork(
            config.stateDim,
            hiddenLayers,
            config.dropoutRate,
            config.hiddenActivation
        );
    }


    selectAction(state) {
        const actionProbabilities = this.actor.forward(state, true, false); // Inference mode
        const value = this.critic.forward(state, false, false)[0]; // Inference mode

        // Sample action from probabilities (categorical distribution)
        let action = 0;
        let cumulativeProbability = 0;
        const rand = Math.random();
        for (let i = 0; i < actionProbabilities.length; i++) {
            cumulativeProbability += actionProbabilities[i];
            if (rand < cumulativeProbability) {
                action = i;
                break;
            }
        }

        const logProb = Math.log(actionProbabilities[action]);

        return { action, logProb, value };
    }

    remember(state, action, reward, nextState, done, logProb, value) {
        this.experiences.push({ state, action, reward, nextState, done, logProb, value });
    }

    update() {
        if (this.experiences.length === 0) {
            console.log("No experiences to update PPO agent.");
            return;
        }

        // 1. Calculate Returns (GAE - Generalized Advantage Estimation)
        const states = this.experiences.map(exp => exp.state);
        const actions = this.experiences.map(exp => exp.action);
        const rewards = this.experiences.map(exp => exp.reward);
        const nextStates = this.experiences.map(exp => exp.nextState);
        const dones = this.experiences.map(exp => exp.done);
        const oldLogProbs = this.experiences.map(exp => exp.logProb);
        const oldValues = this.experiences.map(exp => exp.value);

        const advantages = [];
        const returns = [];

        let lastAdvantage = 0;
        for (let i = this.experiences.length - 1; i >= 0; i--) {
            const reward = rewards[i];
            const done = dones[i];
            const value = oldValues[i];
            const nextValue = i + 1 < this.experiences.length ? oldValues[i + 1] : (done ? 0 : this.critic.forward(nextStates[i], false, false)[0]);

            const delta = reward + this.config.gamma * nextValue * (done ? 0 : 1) - value;
            lastAdvantage = delta + this.config.gamma * this.config.lambda * (done ? 0 : 1) * lastAdvantage;
            advantages.unshift(lastAdvantage);
            returns.unshift(lastAdvantage + value);
        }

        // Normalize advantages
        const meanAdvantage = advantages.reduce((sum, val) => sum + val, 0) / advantages.length;
        const stdAdvantage = Math.sqrt(advantages.map(val => (val - meanAdvantage) ** 2).reduce((sum, val) => sum + val, 0) / advantages.length) + 1e-8;
        const normalizedAdvantages = advantages.map(val => (val - meanAdvantage) / stdAdvantage);

        // 2. PPO Epochs
        for (let epoch = 0; epoch < this.config.ppoEpochs; epoch++) {
            // Shuffle indices for mini-batching
            const indices = Array.from({ length: this.experiences.length }, (_, i) => i);
            for (let i = indices.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [indices[i], indices[j]] = [indices[j], indices[i]]; // Swap
            }

            for (let i = 0; i < indices.length; i += this.config.miniBatchSize) {
                const miniBatchIndices = indices.slice(i, i + this.config.miniBatchSize);

                const miniBatchStates = miniBatchIndices.map(idx => states[idx]);
                const miniBatchActions = miniBatchIndices.map(idx => actions[idx]);
                const miniBatchOldLogProbs = miniBatchIndices.map(idx => oldLogProbs[idx]);
                const miniBatchNormalizedAdvantages = miniBatchIndices.map(idx => normalizedAdvantages[idx]);
                const miniBatchReturns = miniBatchIndices.map(idx => returns[idx]);

                // --- Actor Update ---
                // Calculate current action probabilities and log probabilities
                const currentActionProbabilities = miniBatchStates.map(s => this.actor.forward(s, true, true)); // Training mode
                const currentLogProbs = currentActionProbabilities.map((probs, idx) => Math.log(probs[miniBatchActions[idx]]));

                // Calculate ratio (new_policy / old_policy)
                const ratios = currentLogProbs.map((lp, idx) => Math.exp(lp - miniBatchOldLogProbs[idx]));

                // Calculate clipped surrogate objective
                let actorLoss = 0;
                const epsilon = this.config.epsilon;
                for (let j = 0; j < ratios.length; j++) {
                    const ratio = ratios[j];
                    const advantage = miniBatchNormalizedAdvantages[j];

                    const term1 = ratio * advantage;
                    const term2 = Math.min(Math.max(ratio, 1 - epsilon), 1 + epsilon) * advantage;
                    actorLoss += -Math.min(term1, term2); // PPO is a maximization problem, so we negate for gradient descent
                }
                actorLoss /= ratios.length; // Average loss over mini-batch

                // Backpropagate actor loss
                const actorOutputGradient = currentActionProbabilities.map((probs, idx) => {
                    const actionTaken = miniBatchActions[idx];
                    const grad = Array(probs.length).fill(0);
                    const ratio = ratios[idx];
                    const advantage = miniBatchNormalizedAdvantages[idx];

                    if (ratio > 1 + epsilon) {
                        grad[actionTaken] = -advantage;
                    } else if (ratio < 1 - epsilon) {
                        grad[actionTaken] = advantage;
                    } else {
                        grad[actionTaken] = -advantage;
                    }
                    return grad;
                }).flat();

                this.actor.zeroGrad();
                this.actor.backward(actorOutputGradient);
                this.actor.applyGradients(this.config.learningRateActor, 0.9, 0.999, 1e-8, this.config.l2RegularizationRate, this.config.clipNorm);

                // --- Critic Update ---
                // Calculate current value estimates
                const currentValues = miniBatchStates.map(s => this.critic.forward(s, false, true)[0]); // Training mode

                // Calculate critic loss (MSE)
                let criticLoss = 0;
                for (let j = 0; j < currentValues.length; j++) {
                    criticLoss += (currentValues[j] - miniBatchReturns[j]) ** 2;
                }
                criticLoss /= currentValues.length; // Average loss over mini-batch

                // Backpropagate critic loss
                const criticOutputGradient = currentValues.map((val, idx) => 2 * (val - miniBatchReturns[idx]));

                this.critic.zeroGrad();
                this.critic.backward(criticOutputGradient);
                this.critic.applyGradients(this.config.learningRateCritic);
            }
        }

        // Clear experiences after update
        this.experiences = [];
    }

}

// --- GUI Logic ---
const canvas = document.getElementById('cartPoleCanvas');
const ctx = canvas.getContext('2d');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const evaluateButton = document.getElementById('evaluateButton');
const resetButton = document.getElementById('resetButton');
const infoBox = document.getElementById('infoBox');
const applyConfigButton = document.getElementById('applyConfigButton');

const configInputs = {
    learningRateActor: { input: document.getElementById('learningRateActor'), valueDisplay: document.getElementById('learningRateActorValue') },
    learningRateCritic: { input: document.getElementById('learningRateCritic'), valueDisplay: document.getElementById('learningRateCriticValue') },
    gamma: { input: document.getElementById('gamma'), valueDisplay: document.getElementById('gammaValue') },
    lambda: { input: document.getElementById('lambda'), valueDisplay: document.getElementById('lambdaValue') },
    epsilon: { input: document.getElementById('epsilon'), valueDisplay: document.getElementById('epsilonValue') },
    ppoEpochs: { input: document.getElementById('ppoEpochs'), valueDisplay: document.getElementById('ppoEpochsValue') },
    miniBatchSize: { input: document.getElementById('miniBatchSize'), valueDisplay: document.getElementById('miniBatchSizeValue') },
    dropoutRate: { input: document.getElementById('dropoutRate'), valueDisplay: document.getElementById('dropoutRateValue') },
    l2RegularizationRate: { input: document.getElementById('l2RegularizationRate'), valueDisplay: document.getElementById('l2RegularizationRateValue') },
    clipNorm: { input: document.getElementById('clipNorm'), valueDisplay: document.getElementById('clipNormValue') },
    hiddenActivation: { input: document.getElementById('hiddenActivation') },
};

const CART_WIDTH = 80;
const CART_HEIGHT = 20;
const POLE_LENGTH = 100;
const AXLE_RADIUS = 5;
const WHEEL_RADIUS = 10;
const TRACK_Y = canvas.height - 50;

let agent;
let env;
let animationFrameId;
let isTraining = false;
let isEvaluating = false;
let episodeCount = 0;
let totalSteps = 0;
let totalReward = 0;

const config = {
    stateDim: 4,
    actionDim: 2,
    learningRateActor: 0.0003,
    learningRateCritic: 0.000001,
    gamma: 0.995,
    lambda: 0.9,
    epsilon: 0.2,
    ppoEpochs: 20,
    miniBatchSize: 128,
    dropoutRate: 0.0,
    l2RegularizationRate: 0.00001,
    clipNorm: 1.0,
    hiddenActivation: 'leakyRelu',
};

function loadConfigToUI() {
    for (const key in configInputs) {
        if (configInputs.hasOwnProperty(key)) {
            const inputGroup = configInputs[key];
            const inputElement = inputGroup.input;
            const valueDisplayElement = inputGroup.valueDisplay;

            if (inputElement.type === 'range' || inputElement.type === 'number') {
                inputElement.value = config[key];
                if (valueDisplayElement) {
                    valueDisplayElement.textContent = parseFloat(config[key]).toFixed(5); // Display with precision
                }
            } else if (inputElement.tagName === 'SELECT') {
                inputElement.value = config[key];
            }
        }
    }
}

function applyConfigFromUI() {
    for (const key in configInputs) {
        if (configInputs.hasOwnProperty(key)) {
            const inputGroup = configInputs[key];
            const inputElement = inputGroup.input;

            if (inputElement.type === 'range' || inputElement.type === 'number') {
                config[key] = parseFloat(inputElement.value);
            } else if (inputElement.tagName === 'SELECT') {
                config[key] = inputElement.value;
            }
        }
    }
    logInfo('Applying new configuration...');
    initAgent(); // Re-initialize agent with new config
}

function initAgent() {
    agent = new PPOAgent(config);
    env = new CartPoleEnvironment();
    episodeCount = 0;
    totalSteps = 0;
    totalReward = 0;
    logInfo('Agent and Environment Initialized. Ready to train.');
    updateButtons(false, false, true);
    loadConfigToUI(); // Load current config to UI

    // Add event listeners for sliders to update display
    for (const key in configInputs) {
        if (configInputs.hasOwnProperty(key)) {
            const inputGroup = configInputs[key];
            const inputElement = inputGroup.input;
            const valueDisplayElement = inputGroup.valueDisplay;

            if (inputElement.type === 'range' && valueDisplayElement) {
                inputElement.addEventListener('input', () => {
                    valueDisplayElement.textContent = parseFloat(inputElement.value).toFixed(5);
                });
            }
        }
    }
}

function logInfo(message) {
    infoBox.innerHTML += message + '\n';
    infoBox.scrollTop = infoBox.scrollHeight;
}

function updateButtons(training, evaluating, canEvaluate) {
    startButton.disabled = training || evaluating;
    stopButton.disabled = !(training || evaluating);
    evaluateButton.disabled = training || evaluating || !canEvaluate;
    resetButton.disabled = training || evaluating;
    applyConfigButton.disabled = training || evaluating; // Disable apply config during training/evaluation
}

function drawCartPole(state) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const cartX = canvas.width / 2 + state[0] * 100; // Scale position
    const cartY = TRACK_Y - CART_HEIGHT / 2;

    // Draw track
    ctx.beginPath();
    ctx.moveTo(0, TRACK_Y);
    ctx.lineTo(canvas.width, TRACK_Y);
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw cart
    ctx.fillStyle = '#4682B4'; // Steelblue
    ctx.fillRect(cartX - CART_WIDTH / 2, cartY - CART_HEIGHT / 2, CART_WIDTH, CART_HEIGHT);

    // Draw wheels
    ctx.fillStyle = '#333';
    ctx.beginPath();
    ctx.arc(cartX - CART_WIDTH / 4, cartY + CART_HEIGHT / 2, WHEEL_RADIUS, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cartX + CART_WIDTH / 4, cartY + CART_HEIGHT / 2, WHEEL_RADIUS, 0, Math.PI * 2);
    ctx.fill();

    // Draw pole
    const poleX = cartX;
    const poleY = cartY - CART_HEIGHT / 2;
    const poleEndX = poleX + POLE_LENGTH * Math.sin(state[2]);
    const poleEndY = poleY - POLE_LENGTH * Math.cos(state[2]);

    ctx.beginPath();
    ctx.moveTo(poleX, poleY);
    ctx.lineTo(poleEndX, poleEndY);
    ctx.strokeStyle = '#A52A2A'; // Brown
    ctx.lineWidth = 6;
    ctx.stroke();

    // Draw axle
    ctx.beginPath();
    ctx.arc(poleX, poleY, AXLE_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = '#333';
    ctx.fill();
}

async function runEpisode(mode) {
    let state = env.reset();
    let done = false;
    let episodeReward = 0;
    let stepsInEpisode = 0;

    while (!done && stepsInEpisode < env.maxSteps) {
        const { action, logProb, value } = agent.selectAction(state);
        const { state: nextState, reward, done: envDone } = env.step(action);

        if (mode === 'train') {
            agent.remember(state, action, reward, nextState, envDone, logProb, value);
        }

        state = nextState;
        episodeReward += reward;
        done = envDone;
        stepsInEpisode++;

        drawCartPole(state);
        await new Promise(resolve => requestAnimationFrame(resolve)); // Visual delay
    }

    if (mode === 'train') {
        agent.update();
    }

    episodeCount++;
    totalSteps += stepsInEpisode;
    totalReward += episodeReward;

    logInfo(`Episode ${episodeCount} (${mode}): Reward = ${episodeReward}, Steps = ${stepsInEpisode}`);

    return episodeReward;
}

async function trainAgent() {
    isTraining = true;
    isEvaluating = false;
    updateButtons(true, false, false);
    logInfo('Starting training...');

    let consecutiveSolvedEpisodes = 0;
    const solvedRewardThreshold = env.maxSteps;
    const consecutiveSolvedEpisodesThreshold = 5;

    while (isTraining) {
        const episodeReward = await runEpisode('train');

        if (episodeReward >= solvedRewardThreshold) {
            consecutiveSolvedEpisodes++;
            if (consecutiveSolvedEpisodes >= consecutiveSolvedEpisodesThreshold) {
                logInfo(`Agent solved the environment for ${consecutiveSolvedEpisodesThreshold} consecutive episodes. Training complete.`);
                isTraining = false;
                updateButtons(false, false, true);
                break;
            }
        } else {
            consecutiveSolvedEpisodes = 0;
        }

        if (!isTraining) break; // Check if stopped by user
    }
    logInfo('Training finished.');
}

async function evaluateAgent() {
    isEvaluating = true;
    isTraining = false;
    updateButtons(false, true, false);
    logInfo('Starting evaluation...');

    const numEvaluationEpisodes = 10; // Run 10 evaluation episodes
    let evaluationRewards = [];

    for (let i = 0; i < numEvaluationEpisodes; i++) {
        if (!isEvaluating) break; // Allow stopping evaluation
        const episodeReward = await runEpisode('evaluate');
        evaluationRewards.push(episodeReward);
    }

    const averageReward = evaluationRewards.reduce((sum, r) => sum + r, 0) / evaluationRewards.length;
    logInfo(`Evaluation finished. Average Reward over ${evaluationRewards.length} episodes: ${averageReward.toFixed(2)}`);
    isEvaluating = false;
    updateButtons(false, false, true);
}

function stopAgent() {
    isTraining = false;
    isEvaluating = false;
    cancelAnimationFrame(animationFrameId);
    logInfo('Simulation stopped.');
    updateButtons(false, false, true);
}

function resetSimulation() {
    stopAgent();
    initAgent();
    drawCartPole(env.reset()); // Draw initial state
    infoBox.innerHTML = ''; // Clear info box
    logInfo('Simulation reset.');
}

// Event Listeners
startButton.addEventListener('click', trainAgent);
stopButton.addEventListener('click', stopAgent);
evaluateButton.addEventListener('click', evaluateAgent);
resetButton.addEventListener('click', resetSimulation);
applyConfigButton.addEventListener('click', applyConfigFromUI);

// Initial setup
initAgent();
drawCartPole(env.reset());
updateButtons(false, false, false); // Initially, only start and reset are enabled
resetButton.disabled = false; // Reset is always enabled
startButton.disabled = false; // Start is enabled after init
