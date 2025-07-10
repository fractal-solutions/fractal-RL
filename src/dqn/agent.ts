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

    public sample(batchSize: number): Experience[] | undefined {
        if (this.size() < batchSize) {
            return undefined;
        }

        const batch: Experience[] = [];
        const indices = new Set<number>();
        while (indices.size < batchSize) {
            const randomIndex = Math.floor(Math.random() * this.size());
            if (!indices.has(randomIndex)) {
                indices.add(randomIndex);
                batch.push(this.buffer[randomIndex]!);
            }
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
        const experiences = this.replayBuffer.sample(this.config.batchSize);
        if (!experiences) {
            return; // Not enough experiences to learn
        }

        const states = experiences.map(exp => exp.state);
        const actions = experiences.map(exp => exp.action);
        const rewards = experiences.map(exp => exp.reward);
        const nextStates = experiences.map(exp => exp.nextState);
        const dones = experiences.map(exp => exp.done);

        const currentQValues = states.map(s => this.qNetwork.forward(s, true));
        const nextQValuesTarget = nextStates.map(ns => this.targetQNetwork.forward(ns, false));

        const targetQValues: number[][] = currentQValues.map(arr => [...arr]);

        for (let i = 0; i < this.config.batchSize; i++) {
            const action = actions[i]!;
            const reward = rewards[i]!;
            const done = dones[i]!;
            const nextQValue = nextQValuesTarget[i]!;

            const maxNextQ = Math.max(...nextQValue);
            const newQ = reward + this.config.gamma * maxNextQ * (done ? 0 : 1);
            targetQValues[i]![action] = newQ;
        }

        this.qNetwork.zeroGrad();
        for (let i = 0; i < this.config.batchSize; i++) {
            const action = actions[i]!;
            const prediction = currentQValues[i]![action]!;
            const target = targetQValues[i]![action]!;
            const grad = Array(this.config.actionDim).fill(0);
            grad[action] = 2 * (prediction - target) / this.config.batchSize;
            this.qNetwork.backward(grad);
        }
        this.qNetwork.applyGradients(this.config.learningRate, 0.9, 0.999, 1e-8, this.config.l2RegularizationRate, this.config.clipNorm);

        this.learnStepCounter++;

        if (this.learnStepCounter % this.config.targetUpdateFrequency === 0) {
            this.updateTargetNetwork();
        }

        this.epsilon = Math.max(this.config.epsilonEnd, this.epsilon * this.config.epsilonDecay);
    }
}