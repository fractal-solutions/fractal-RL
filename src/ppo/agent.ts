// src/ppo/agent.ts
import { ActorNetwork, CriticNetwork } from './network';

interface PPOAgentConfig {
    stateDim: number;
    actionDim: number;
    learningRateActor: number;
    learningRateCritic: number;
    gamma: number; // Discount factor
    lambda: number; // GAE parameter
    epsilon: number; // Clip parameter
    ppoEpochs: number;
    miniBatchSize: number;
    dropoutRate?: number; // Optional: Dropout rate for regularization
    l2RegularizationRate?: number; // Optional: L2 regularization rate
    clipNorm?: number; // Optional: Gradient clipping norm
    hiddenActivation?: string; // Optional: Name of the hidden activation function
}

// Represents a single experience tuple
interface Experience {
    state: number[];
    action: number;
    reward: number;
    nextState: number[];
    done: boolean;
    logProb: number; // Log probability of the action taken by the old policy
    value: number; // Value estimate of the state by the old critic
}

export class PPOAgent {
    private actor: ActorNetwork;
    private critic: CriticNetwork;
    private config: PPOAgentConfig;
    private experiences: Experience[]; // Buffer to store experiences

    constructor(config: PPOAgentConfig) {
        this.config = config;
        this.actor = new ActorNetwork(config.stateDim, config.actionDim, [64, 64], config.dropoutRate, config.hiddenActivation);
        this.critic = new CriticNetwork(config.stateDim, [64, 64], config.dropoutRate, config.hiddenActivation);
        this.experiences = [];
    }

    public selectAction(state: number[]): { action: number, logProb: number, value: number } {
        const actionProbabilities = this.actor.forward(state, false); // Inference mode
        const value = this.critic.forward(state, false)[0]!;

        let action = 0;
        let cumulativeProbability = 0;
        const rand = Math.random();
        for (let i = 0; i < actionProbabilities.length; i++) {
            cumulativeProbability += actionProbabilities[i]!;
            if (rand < cumulativeProbability) {
                action = i;
                break;
            }
        }

        const logProb = Math.log(actionProbabilities[action]! + 1e-10); // Add epsilon for numerical stability

        return { action, logProb, value };
    }

    public remember(state: number[], action: number, reward: number, nextState: number[], done: boolean, logProb: number, value: number): void {
        this.experiences.push({ state, action, reward, nextState, done, logProb, value });
    }

    public update(): void {
        if (this.experiences.length === 0) {
            return;
        }

        const states = this.experiences.map(exp => exp.state);
        const actions = this.experiences.map(exp => exp.action);
        const rewards = this.experiences.map(exp => exp.reward);
        const nextStates = this.experiences.map(exp => exp.nextState);
        const dones = this.experiences.map(exp => exp.done);
        const oldLogProbs = this.experiences.map(exp => exp.logProb);
        const oldValues = this.experiences.map(exp => exp.value);

        const advantages: number[] = [];
        const returns: number[] = [];
        let lastAdvantage = 0;

        for (let i = this.experiences.length - 1; i >= 0; i--) {
            const done = dones[i]!;
            const reward = rewards[i]!;
            const value = oldValues[i]!;
            const nextValue = (i === this.experiences.length - 1) ? (done ? 0 : this.critic.forward(nextStates[i]!, false)[0]!) : oldValues[i + 1]!;
            
            const delta = reward + this.config.gamma * nextValue * (done ? 0 : 1) - value;
            lastAdvantage = delta + this.config.gamma * this.config.lambda * (done ? 0 : 1) * lastAdvantage;
            advantages.unshift(lastAdvantage);
            returns.unshift(lastAdvantage + value);
        }

        const meanAdvantage = advantages.reduce((sum, val) => sum + val, 0) / advantages.length;
        const stdAdvantage = Math.sqrt(advantages.map(val => (val - meanAdvantage) ** 2).reduce((sum, val) => sum + val, 0) / advantages.length) + 1e-8;
        const normalizedAdvantages = advantages.map(val => (val - meanAdvantage) / stdAdvantage);

        for (let epoch = 0; epoch < this.config.ppoEpochs; epoch++) {
            const indices = Array.from({ length: this.experiences.length }, (_, i) => i);
            // Shuffle indices
            for (let i = indices.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [indices[i], indices[j]] = [indices[j]!, indices[i]!];
            }

            for (let i = 0; i < indices.length; i += this.config.miniBatchSize) {
                const miniBatchIndices = indices.slice(i, i + this.config.miniBatchSize);

                const miniBatchStates = miniBatchIndices.map(idx => states[idx]!);
                const miniBatchActions = miniBatchIndices.map(idx => actions[idx]!);
                const miniBatchOldLogProbs = miniBatchIndices.map(idx => oldLogProbs[idx]!);
                const miniBatchNormalizedAdvantages = miniBatchIndices.map(idx => normalizedAdvantages[idx]!);
                const miniBatchReturns = miniBatchIndices.map(idx => returns[idx]!);

                // Actor Update
                const currentActionProbabilities = miniBatchStates.map(s => this.actor.forward(s, true));
                const currentLogProbs = currentActionProbabilities.map((probs, idx) => Math.log(probs[miniBatchActions[idx]!]! + 1e-10));
                const ratios = currentLogProbs.map((lp, idx) => Math.exp(lp - miniBatchOldLogProbs[idx]!));

                this.actor.zeroGrad();
                for (let j = 0; j < ratios.length; j++) {
                    const ratio = ratios[j]!;
                    const advantage = miniBatchNormalizedAdvantages[j]!;
                    const term1 = ratio * advantage;
                    const term2 = Math.min(Math.max(ratio, 1 - this.config.epsilon), 1 + this.config.epsilon) * advantage;
                    const actorLoss = -Math.min(term1, term2) / ratios.length;

                    const grad = Array(this.config.actionDim).fill(0);
                    grad[miniBatchActions[j]!] = -actorLoss; // Simplified gradient
                    this.actor.backward(grad);
                }
                this.actor.applyGradients(this.config.learningRateActor, 0.9, 0.999, 1e-8, this.config.l2RegularizationRate, this.config.clipNorm);

                // Critic Update
                const currentValues = miniBatchStates.map(s => this.critic.forward(s, true)[0]!);
                this.critic.zeroGrad();
                for (let j = 0; j < currentValues.length; j++) {
                    const criticLoss = (currentValues[j]! - miniBatchReturns[j]!) ** 2 / currentValues.length;
                    const grad = [2 * (currentValues[j]! - miniBatchReturns[j]!) / currentValues.length];
                    this.critic.backward(grad);
                }
                this.critic.applyGradients(this.config.learningRateCritic, 0.9, 0.999, 1e-8, this.config.l2RegularizationRate, this.config.clipNorm);
            }
        }

        this.experiences = [];
    }
}