// src/dqn/network.ts
// A simple Q-network for DQN.
// Similar to the PPO network, this is a basic implementation.

import { getActivation, ActivationFunction, ActivationDerivative } from '../utils/activations.ts';

export class QNetwork {
    protected weights: number[][][]; // [layer_idx][input_node_idx][output_node_idx]
    protected biases: number[][]; // [layer_idx][output_node_idx]
    protected layerSizes: number[];
    protected hiddenActivation: { func: ActivationFunction, deriv: ActivationDerivative };

    // For backpropagation
    protected activations: number[][]; // Activations for each layer (including input)
    protected zValues: number[][];     // Pre-activation values (sum before activation function)
    protected weightGradients: number[][][];
    protected biasGradients: number[][];

    // For Adam optimizer
    protected mWeights: number[][][]; // First moment estimates for weights
    protected vWeights: number[][][]; // Second moment estimates for weights
    protected mBiases: number[][];    // First moment estimates for biases
    protected vBiases: number[][];    // Second moment estimates for biases
    protected t: number;              // Timestep for Adam

    

    // For Dropout
    protected dropoutMasks: boolean[][];
    protected dropoutRate: number;

    constructor(inputSize: number, outputSize: number, hiddenLayers: number[] = [64, 64], dropoutRate: number = 0.0, hiddenActivationName: string = 'leakyRelu') {
        this.hiddenActivation = getActivation(hiddenActivationName);
        this.layerSizes = [inputSize, ...hiddenLayers, outputSize];
        this.weights = [];
        this.biases = [];
        this.weightGradients = [];
        this.biasGradients = [];
        this.activations = [];
        this.zValues = [];

        // Initialize Adam parameters
        this.mWeights = [];
        this.vWeights = [];
        this.mBiases = [];
        this.vBiases = [];
        this.t = 0;

        

        // Initialize Dropout parameters
        this.dropoutMasks = [];
        this.dropoutRate = dropoutRate;

        for (let i = 0; i < this.layerSizes.length - 1; i++) {
            const inputDim = this.layerSizes[i];
            const outputDim = this.layerSizes[i + 1];

            // Initialize weights (He initialization)
            const layerWeights: number[][] = Array(inputDim).fill(0).map(() =>
                Array(outputDim).fill(0).map(() => (Math.random() * 2 - 1) * Math.sqrt(2 / inputDim))
            );
            this.weights.push(layerWeights);

            // Initialize biases
            const layerBiases: number[] = Array(outputDim).fill(0);
            this.biases.push(layerBiases);

            // Initialize gradients to zeros
            this.weightGradients.push(Array(inputDim).fill(0).map(() => Array(outputDim).fill(0)));
            this.biasGradients.push(Array(outputDim).fill(0));

            // Initialize Adam moment estimates
            this.mWeights.push(Array(inputDim).fill(0).map(() => Array(outputDim).fill(0)));
            this.vWeights.push(Array(inputDim).fill(0).map(() => Array(outputDim).fill(0)));
            this.mBiases.push(Array(outputDim).fill(0));
            this.vBiases.push(Array(outputDim).fill(0));

            
        }
    }

    

    // Forward pass
    public forward(input: number[], isTraining: boolean = true): number[] {
        if (input.length !== this.layerSizes[0]) {
            throw new Error(`Input size mismatch. Expected ${this.layerSizes[0]}, got ${input.length}`);
        }

        this.activations = [];
        this.zValues = [];
        this.normalizedZValues = []; // Reset for each forward pass
        this.dropoutMasks = []; // Reset for each forward pass

        let currentActivations = input;
        this.activations.push(currentActivations); // Store input activations

        for (let i = 0; i < this.weights.length; i++) {
            const nextZValues: number[] = Array(this.layerSizes[i + 1]).fill(0);

            for (let j = 0; j < this.layerSizes[i + 1]; j++) {
                let sum = this.biases[i][j];
                for (let k = 0; k < this.layerSizes[i]; k++) {
                    sum += currentActivations[k] * this.weights[i][k][j];
                }
                nextZValues[j] = sum;
            }
            this.zValues.push(nextZValues); // Store pre-activation values

            // Apply activation function (Leaky ReLU for hidden layers, linear for output)
            if (i < this.weights.length - 1) {
                currentActivations = nextZValues.map(this.hiddenActivation.func);

                // Apply Dropout
                if (isTraining && this.dropoutRate > 0) {
                    const mask: boolean[] = Array(currentActivations.length).fill(false).map(() => Math.random() > this.dropoutRate);
                    this.dropoutMasks.push(mask);
                    currentActivations = currentActivations.map((val, idx) => mask[idx] ? val / (1 - this.dropoutRate) : 0);
                }
            } else {
                currentActivations = nextZValues; // Output layer is linear for Q-values
            }
            this.activations.push(currentActivations); // Store layer activations
        }
        return currentActivations;
    }

    // Backpropagation
    // `outputGradient` is the gradient of the loss with respect to the network's output.
    public backward(outputGradient: number[]): void {
        this.zeroGrad(); // Reset gradients before calculating new ones

        let currentDelta = outputGradient;

        // Iterate backwards through the layers
        for (let i = this.weights.length - 1; i >= 0; i--) {
            const currentLayerActivations = this.activations[i + 1]; // Activations of the current layer
            const prevLayerActivations = this.activations[i];     // Activations of the previous layer (input to current layer)
            const currentZValues = this.zValues[i];

            // Apply derivative of activation function (if not output layer)
            if (i < this.weights.length - 1) { // Hidden layer
                // Backward pass for Dropout
                if (this.dropoutRate > 0 && this.dropoutMasks[i]) {
                    currentDelta = currentDelta.map((d, idx) => this.dropoutMasks[i][idx] ? d / (1 - this.dropoutRate) : 0);
                }

                currentDelta = currentDelta.map((d, idx) => d * this.hiddenActivation.deriv(currentZValues[idx]));
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
            const nextDelta: number[] = Array(this.layerSizes[i]).fill(0);
            for (let j = 0; j < this.layerSizes[i]; j++) { // For each node in the previous layer
                for (let k = 0; k < this.layerSizes[i + 1]; k++) {
                    nextDelta[j] += currentDelta[k] * this.weights[i][j][k];
                }
            }
            currentDelta = nextDelta;
        }
    }

    // Apply gradients using the Adam optimizer with L2 regularization and Gradient Clipping
    public applyGradients(learningRate: number, beta1: number = 0.9, beta2: number = 0.999, epsilon: number = 1e-8, l2RegularizationRate: number = 0.001, clipNorm: number = 1.0): void {
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
    public zeroGrad(): void {
        for (let i = 0; i < this.weightGradients.length; i++) {
            for (let j = 0; j < this.weightGradients[i].length; j++) {
                for (let k = 0; k < this.weightGradients[i][j].length; k++) {
                    this.weightGradients[i][j][k] = 0;
                    this.mWeights[i][j][k] = 0;
                    this.vWeights[i][j][k] = 0;
                }
            }
            for (let j = 0; j < this.biasGradients[i].length; j++) {
                this.biasGradients[i][j] = 0;
                this.mBiases[i][j] = 0;
                this.vBiases[i][j] = 0;
            }
            this.t = 0;
            this.dropoutMasks = []; // Clear dropout masks
        }
    }

    // Method to copy parameters from another QNetwork instance
    public copyParametersFrom(otherNetwork: QNetwork): void {
        if (this.layerSizes.toString() !== otherNetwork.layerSizes.toString()) {
            throw new Error("Cannot copy parameters: network architectures do not match.");
        }
        // Deep copy weights
        this.weights = otherNetwork.weights.map(layer => layer.map(row => [...row]));
        // Deep copy biases
        this.biases = otherNetwork.biases.map(layer => [...layer]);
    }
}