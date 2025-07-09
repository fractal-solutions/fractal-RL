// src/dqn/agent.ts
import { QNetwork } from './network';

interface DQNAgentConfig {
    stateDim: number;
    actionDim: number;
    learningRate: number;
    gamma: number; // Discount factor
    epsilonStart: number;
    epsilonEnd: number;
    epsilonDecay: number;
    batchSize: number;
    targetUpdateFrequency: number;
    replayBufferSize: number;
    dropoutRate: number; // Dropout rate for regularization
    l2RegularizationRate: number; // L2 regularization rate
    clipNorm: number; // Gradient clipping norm
    hiddenActivation?: string; // Optional: Name of the hidden activation function (e.g., 'relu', 'leakyRelu')
}

// Represents a single experience tuple
interface Experience {
    state: number[];
    action: number;
    reward: number;
    nextState: number[];
    done: boolean;
}

class ReplayBuffer {
    private buffer: Experience[];
    private capacity: number;
    private position: number;

    constructor(capacity: number) {
        this.capacity = capacity;
        this.buffer = new Array<Experience>(capacity);
        this.position = 0;
    }

    public push(state: number[], action: number, reward: number, nextState: number[], done: boolean): void {
        this.buffer[this.position] = { state, action, reward, nextState, done };
        this.position = (this.position + 1) % this.capacity;
    }

    public sample(batchSize: number): Experience[] {
        const actualSize = Math.min(this.size(), this.capacity);
        const batch: Experience[] = [];
        for (let i = 0; i < batchSize; i++) {
            const randomIndex = Math.floor(Math.random() * actualSize);
            batch.push(this.buffer[randomIndex]);
        }
        return batch;
    }

    public size(): number {
        return Math.min(this.position, this.capacity);
    }
}

export class DQNAgent {
    private qNetwork: QNetwork;
    private targetQNetwork: QNetwork;
    private config: DQNAgentConfig;
    private epsilon: number;
    private learnStepCounter: number;
    private replayBuffer: ReplayBuffer;

    constructor(config: DQNAgentConfig) {
        this.config = config;
        this.qNetwork = new QNetwork(config.stateDim, config.actionDim, undefined, config.dropoutRate, config.hiddenActivation);
        this.targetQNetwork = new QNetwork(config.stateDim, config.actionDim, undefined, config.dropoutRate, config.hiddenActivation);
        this.updateTargetNetwork(); // Initialize target network
        this.epsilon = config.epsilonStart;
        this.learnStepCounter = 0;
        this.replayBuffer = new ReplayBuffer(config.replayBufferSize);
    }

    private updateTargetNetwork(): void {
        this.targetQNetwork.copyParametersFrom(this.qNetwork);
    }

    public selectAction(state: number[]): number {
        if (Math.random() < this.epsilon) {
            return Math.floor(Math.random() * this.config.actionDim); // Explore
        } else {
            const qValues = this.qNetwork.forward(state, false); // Inference mode
            return qValues.indexOf(Math.max(...qValues)); // Exploit
        }
    }

    public remember(state: number[], action: number, reward: number, nextState: number[], done: boolean): void {
        this.replayBuffer.push(state, action, reward, nextState, done);
    }

    public learn(): void {
        if (this.replayBuffer.size() < this.config.batchSize) {
            return; // Not enough experiences in buffer to learn
        }

        const experiences = this.replayBuffer.sample(this.config.batchSize);

        const states = experiences.map(exp => exp.state);
        const actions = experiences.map(exp => exp.action);
        const rewards = experiences.map(exp => exp.reward);
        const nextStates = experiences.map(exp => exp.nextState);
        const dones = experiences.map(exp => exp.done);

        // Calculate Q-targets
        const currentQValues = states.map(s => this.qNetwork.forward(s, true)); // Training mode
        const nextQValuesTarget = nextStates.map(ns => this.targetQNetwork.forward(ns, false)); // Inference mode for target network

        const targetQValues: number[][] = currentQValues.map(arr => [...arr]); // Deep copy

        for (let i = 0; i < this.config.batchSize; i++) {
            const maxNextQ = Math.max(...nextQValuesTarget[i]);
            const newQ = rewards[i] + this.config.gamma * maxNextQ * (dones[i] ? 0 : 1);
            targetQValues[i][actions[i]] = newQ;
        }

        // Calculate MSE loss and backpropagate
        let loss = 0;
        const outputGradients: number[][] = [];

        for (let i = 0; i < this.config.batchSize; i++) {
            const prediction = currentQValues[i][actions[i]];
            const target = targetQValues[i][actions[i]];
            loss += (prediction - target) ** 2; // MSE loss

            // Gradient of MSE w.r.t. prediction: 2 * (prediction - target)
            const grad = Array(this.config.actionDim).fill(0);
            grad[actions[i]] = 2 * (prediction - target);
            outputGradients.push(grad);
        }
        loss /= this.config.batchSize; // Average loss

        // Backpropagate gradients for each sample in the batch
        // This is a simplified batch backpropagation. In a real scenario, you'd sum gradients
        // or use a more efficient batching mechanism within the network's backward pass.
        this.qNetwork.zeroGrad();
        for (let i = 0; i < this.config.batchSize; i++) {
            this.qNetwork.backward(outputGradients[i]);
        }
        this.qNetwork.applyGradients(this.config.learningRate, 0.9, 0.999, 1e-8, this.config.l2RegularizationRate, this.config.clipNorm);

        this.learnStepCounter++;

        // Periodically update the target network
        if (this.learnStepCounter % this.config.targetUpdateFrequency === 0) {
            console.log("Updating target Q-network.");
            this.updateTargetNetwork();
        }

        // Decay epsilon
        this.epsilon = Math.max(this.config.epsilonEnd, this.epsilon * this.config.epsilonDecay);
    }
}