// src/ppo/network.ts
// A very basic neural network implementation from scratch, including forward and backward passes.

import { getActivation, type ActivationFunction, type ActivationDerivative } from '../utils/activations';

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
    protected dropoutMasks: (boolean[] | undefined)[];
    protected dropoutRate: number;

    constructor(layerSizes: number[], dropoutRate: number = 0.0, hiddenActivationName: string = 'leakyRelu') {
        const activation = getActivation(hiddenActivationName);
        if (!activation) {
            throw new Error(`Activation function '${hiddenActivationName}' not found.`);
        }
        this.hiddenActivation = activation;
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
            const inputSize = layerSizes[i]!;
            const outputSize = layerSizes[i + 1]!;

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
        const max = Math.max(...inputs);
        const exp = inputs.map(val => Math.exp(val - max));
        const sumExp = exp.reduce((a, b) => a + b, 0);
        return exp.map(e => e / sumExp);
    }

    // Forward pass through the network
    public forward(input: number[], isTraining: boolean = true, isActor: boolean = false): number[] {
        if (input.length !== this.layerSizes[0]) {
            throw new Error(`Input size mismatch. Expected ${this.layerSizes[0]}, got ${input.length}`);
        }

        this.activations = [];
        this.zValues = [];
        this.dropoutMasks = [];

        let currentActivations = input;
        this.activations.push(currentActivations);

        for (let i = 0; i < this.weights.length; i++) {
            const nextZValues: number[] = Array(this.layerSizes[i + 1]!).fill(0);

            for (let j = 0; j < this.layerSizes[i + 1]!; j++) {
                let sum = this.biases[i]![j]!;
                for (let k = 0; k < this.layerSizes[i]!; k++) {
                    sum += currentActivations[k]! * this.weights[i]![k]![j]!;
                }
                nextZValues[j] = sum;
            }
            this.zValues.push(nextZValues);

            if (i < this.weights.length - 1) {
                currentActivations = nextZValues.map(this.hiddenActivation.func);
                if (isTraining && this.dropoutRate > 0) {
                    const mask: boolean[] = Array(currentActivations.length).fill(false).map(() => Math.random() > this.dropoutRate);
                    this.dropoutMasks.push(mask);
                    currentActivations = currentActivations.map((val, idx) => mask[idx]! ? val / (1 - this.dropoutRate) : 0);
                } else {
                    this.dropoutMasks.push(undefined);
                }
            } else {
                if (isActor) {
                    currentActivations = this.softmax(nextZValues);
                } else {
                    currentActivations = nextZValues;
                }
            }
            this.activations.push(currentActivations);
        }
        return currentActivations;
    }

    // Backpropagation
    public backward(outputGradient: number[], isActor: boolean = false): void {
        let currentDelta = outputGradient;

        for (let i = this.weights.length - 1; i >= 0; i--) {
            const prevLayerActivations = this.activations[i]!;
            const currentZValues = this.zValues[i]!;

            if (i < this.weights.length - 1) {
                const mask = this.dropoutMasks[i];
                if (mask) {
                    currentDelta = currentDelta.map((d, idx) => mask[idx]! ? d / (1 - this.dropoutRate) : 0);
                }
                currentDelta = currentDelta.map((d, idx) => d * this.hiddenActivation.deriv(currentZValues[idx]!));
            } else if (isActor) {
                // The gradient for softmax is handled in the PPO agent's update logic
            }

            for (let j = 0; j < this.biasGradients[i]!.length; j++) {
                this.biasGradients[i]![j]! += currentDelta[j]!;
            }

            for (let j = 0; j < this.weightGradients[i]!.length; j++) {
                for (let k = 0; k < this.weightGradients[i]![j]!.length; k++) {
                    this.weightGradients[i]![j]![k]! += prevLayerActivations[j]! * currentDelta[k]!;
                }
            }

            if (i > 0) {
                const nextDelta: number[] = Array(this.layerSizes[i]!).fill(0);
                for (let j = 0; j < this.layerSizes[i]!; j++) {
                    for (let k = 0; k < this.layerSizes[i + 1]!; k++) {
                        nextDelta[j]! += currentDelta[k]! * this.weights[i]![j]![k]!;
                    }
                }
                currentDelta = nextDelta;
            }
        }
    }

    // Apply gradients using the Adam optimizer
    public applyGradients(learningRate: number, beta1: number = 0.9, beta2: number = 0.999, epsilon: number = 1e-8, l2RegularizationRate?: number, clipNorm?: number): void {
        this.t++;
        const lr_t = learningRate * Math.sqrt(1 - Math.pow(beta2, this.t)) / (1 - Math.pow(beta1, this.t));

        let totalGradNorm = 0;
        if (clipNorm) {
            for (let i = 0; i < this.weights.length; i++) {
                for (let j = 0; j < this.weights[i]!.length; j++) {
                    for (let k = 0; k < this.weights[i]![j]!.length; k++) {
                        totalGradNorm += Math.pow(this.weightGradients[i]![j]![k]!, 2);
                    }
                }
                for (let j = 0; j < this.biases[i]!.length; j++) {
                    totalGradNorm += Math.pow(this.biasGradients[i]![j]!, 2);
                }
            }
            totalGradNorm = Math.sqrt(totalGradNorm);
        }

        const clipFactor = (clipNorm && totalGradNorm > clipNorm) ? clipNorm / totalGradNorm : 1.0;

        for (let i = 0; i < this.weights.length; i++) {
            for (let j = 0; j < this.weights[i]!.length; j++) {
                for (let k = 0; k < this.weights[i]![j]!.length; k++) {
                    const clippedWeightGrad = this.weightGradients[i]![j]![k]! * clipFactor;
                    const weightDecayGrad = (l2RegularizationRate ?? 0) * this.weights[i]![j]![k]!;
                    
                    this.mWeights[i]![j]![k] = beta1 * this.mWeights[i]![j]![k]! + (1 - beta1) * (clippedWeightGrad + weightDecayGrad);
                    this.vWeights[i]![j]![k] = beta2 * this.vWeights[i]![j]![k]! + (1 - beta2) * Math.pow((clippedWeightGrad + weightDecayGrad), 2);

                    const mHat = this.mWeights[i]![j]![k]!;
                    const vHat = this.vWeights[i]![j]![k]!;

                    this.weights[i]![j]![k]! -= lr_t * mHat / (Math.sqrt(vHat) + epsilon);
                }
            }
            for (let j = 0; j < this.biases[i]!.length; j++) {
                const clippedBiasGrad = this.biasGradients[i]![j]! * clipFactor;

                this.mBiases[i]![j] = beta1 * this.mBiases[i]![j]! + (1 - beta1) * clippedBiasGrad;
                this.vBiases[i]![j] = beta2 * this.vBiases[i]![j]! + (1 - beta2) * Math.pow(clippedBiasGrad, 2);

                const mHat = this.mBiases[i]![j]!;
                const vHat = this.vBiases[i]![j]!;

                this.biases[i]![j]! -= lr_t * mHat / (Math.sqrt(vHat) + epsilon);
            }
        }
    }

    // Reset gradients to zero
    public zeroGrad(): void {
        for (let i = 0; i < this.weightGradients.length; i++) {
            for (let j = 0; j < this.weightGradients[i]!.length; j++) {
                this.weightGradients[i]![j]!.fill(0);
            }
            this.biasGradients[i]!.fill(0);
        }
    }

    // Method to copy parameters from another NeuralNetwork instance
    public copyParametersFrom(otherNetwork: NeuralNetwork): void {
        if (this.layerSizes.toString() !== otherNetwork.layerSizes.toString()) {
            throw new Error("Cannot copy parameters: network architectures do not match.");
        }
        this.weights = otherNetwork.weights.map(layer => layer.map(row => [...row]));
        this.biases = otherNetwork.biases.map(layer => [...layer]);
    }
}

export class ActorNetwork extends NeuralNetwork {
    constructor(inputSize: number, outputSize: number, hiddenLayers: number[] = [64, 64], dropoutRate?: number, hiddenActivationName?: string) {
        super([inputSize, ...hiddenLayers, outputSize], dropoutRate, hiddenActivationName);
    }

    public override forward(input: number[], isTraining: boolean = true): number[] {
        return super.forward(input, isTraining, true);
    }

    public override backward(outputGradient: number[]): void {
        super.backward(outputGradient, true);
    }
}

export class CriticNetwork extends NeuralNetwork {
    constructor(inputSize: number, hiddenLayers: number[] = [64, 64], dropoutRate?: number, hiddenActivationName?: string) {
        super([inputSize, ...hiddenLayers, 1], dropoutRate, hiddenActivationName);
    }

    public override forward(input: number[], isTraining: boolean = true): number[] {
        return super.forward(input, isTraining, false);
    }

    public override backward(outputGradient: number[]): void {
        super.backward(outputGradient, false);
    }
}