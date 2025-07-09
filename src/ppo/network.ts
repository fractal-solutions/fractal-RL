
// src/ppo/network.ts
// A very basic neural network implementation from scratch, including forward and backward passes.

import { getActivation, ActivationFunction, ActivationDerivative } from '../utils/activations.ts';

export class NeuralNetwork {
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

    constructor(layerSizes: number[], dropoutRate: number = 0.0, hiddenActivationName: string = 'leakyRelu') {
        this.hiddenActivation = getActivation(hiddenActivationName);
        this.layerSizes = layerSizes;
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

        for (let i = 0; i < layerSizes.length - 1; i++) {
            const inputSize = layerSizes[i];
            const outputSize = layerSizes[i + 1];

            // Initialize weights (He initialization)
            const layerWeights: number[][] = Array(inputSize).fill(0).map(() =>
                Array(outputSize).fill(0).map(() => (Math.random() * 2 - 1) * Math.sqrt(2 / inputSize))
            );
            this.weights.push(layerWeights);

            // Initialize biases to zeros
            const layerBiases: number[] = Array(outputSize).fill(0);
            this.biases.push(layerBiases);

            // Initialize gradients to zeros
            this.weightGradients.push(Array(inputSize).fill(0).map(() => Array(outputSize).fill(0)));
            this.biasGradients.push(Array(outputSize).fill(0));

            // Initialize Adam moment estimates
            this.mWeights.push(Array(inputSize).fill(0).map(() => Array(outputSize).fill(0)));
            this.vWeights.push(Array(inputSize).fill(0).map(() => Array(outputSize).fill(0)));
            this.mBiases.push(Array(outputSize).fill(0));
            this.vBiases.push(Array(outputSize).fill(0));

            
        }
    }

    

    // Softmax activation for output layer of actor (for probabilities)
    protected softmax(inputs: number[]): number[] {
        const exp = inputs.map(Math.exp);
        const sumExp = exp.reduce((a, b) => a + b, 0);
        return exp.map(e => e / sumExp);
    }

    // Derivative of Softmax (simplified for a single output, typically Jacobian for full matrix)
    // For a single output (e.g., loss gradient w.r.t. one output), this is just the output itself.
    // For backprop, we need d(output_j)/d(input_i) which is more complex.
    // Here, we'll assume the gradient coming in is already scaled appropriately for the output.
    // A more rigorous softmax derivative for backprop would involve the Jacobian matrix.
    // For simplicity in this from-scratch implementation, we'll use a common approximation
    // or assume the loss gradient handles the softmax part.
    // Let's use a simplified approach where the gradient is applied to the pre-softmax values.
    // This method is not directly used in the current backward pass, as the loss gradient
    // is assumed to be w.r.t. the pre-softmax outputs for actor loss.

    // Forward pass through the network
    public forward(input: number[], isOutputLayerActor: boolean = false, isTraining: boolean = true): number[] {
        if (input.length !== this.layerSizes[0]) {
            throw new Error(`Input size mismatch. Expected ${this.layerSizes[0]}, got ${input.length}`);
        }

        this.activations = [];
        this.zValues = [];
        this.normalizedZValues = []; // Reset for each forward pass

        let currentActivations = input;
        this.activations.push(currentActivations); // Store input activations

        for (let i = 0; i < this.weights.length; i++) {
            const nextZValues: number[] = Array(this.layerSizes[i + 1]).fill(0);

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
                    const mask: boolean[] = Array(currentActivations.length).fill(false).map(() => Math.random() > this.dropoutRate);
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
    public backward(outputGradient: number[], isOutputLayerActor: boolean = false): void {
        this.zeroGrad(); // Reset gradients before calculating new ones

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
            const nextDelta: number[] = Array(this.layerSizes[i]).fill(0);
            for (let j = 0; j < this.layerSizes[i]; j++) { // For each node in the previous layer
                for (let k = 0; k < this.layerSizes[i + 1]; k++) { // For each node in the current layer
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

    // Method to copy parameters from another NeuralNetwork instance
    public copyParametersFrom(otherNetwork: NeuralNetwork): void {
        if (this.layerSizes.toString() !== otherNetwork.layerSizes.toString()) {
            throw new Error("Cannot copy parameters: network architectures do not match.");
        }
        // Deep copy weights
        this.weights = otherNetwork.weights.map(layer => layer.map(row => [...row]));
        // Deep copy biases
        this.biases = otherNetwork.biases.map(layer => [...layer]);
    }
}

export class ActorNetwork extends NeuralNetwork {
    constructor(inputSize: number, outputSize: number, hiddenLayers: number[] = [64, 64], dropoutRate: number = 0.0, hiddenActivationName: string = 'leakyRelu') {
        super([inputSize, ...hiddenLayers, outputSize], dropoutRate, hiddenActivationName);
    }

    public forward(input: number[]): number[] {
        return super.forward(input, true, true); // Actor network uses softmax for output, training mode
    }

    public backward(outputGradient: number[]): void {
        super.backward(outputGradient, true);
    }
}

export class CriticNetwork extends NeuralNetwork {
    constructor(inputSize: number, hiddenLayers: number[] = [64, 64], dropoutRate: number = 0.0, hiddenActivationName: string = 'leakyRelu') {
        super([inputSize, ...hiddenLayers, 1], dropoutRate, hiddenActivationName); // Critic network outputs a single value
    }

    public forward(input: number[]): number[] {
        return super.forward(input, false, true); // Critic network outputs raw value, training mode
    }

    public backward(outputGradient: number[]): void {
        super.backward(outputGradient, false);
    }
}
