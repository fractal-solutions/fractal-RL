
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
    logProb: number; // Log probability of the action taken by the old policy
    value: number; // Value estimate of the state by the old critic
}

export class PPOAgent {
    private actor: ActorNetwork;
    private critic: CriticNetwork;
    private config: PPOAgentConfig;
    private experiences: Experience[]; // Buffer to store experiences for one trajectory/episode

    constructor(config: PPOAgentConfig) {
        this.config = config;
        this.actor = new ActorNetwork(config.stateDim, config.actionDim, undefined, config.dropoutRate, config.hiddenActivation);
        this.critic = new CriticNetwork(config.stateDim, undefined, config.dropoutRate, config.hiddenActivation);
        this.experiences = [];
    }

    public selectAction(state: number[]): { action: number, logProb: number, value: number } {
        const actionProbabilities = this.actor.forward(state, false); // Inference mode
        const value = this.critic.forward(state, false)[0]; // Inference mode

        // Sample action from probabilities (categorical distribution)
        // For simplicity, we'll use a basic sampling method. In a real scenario,
        // you'd want a more robust categorical sampler.
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

    public remember(state: number[], action: number, reward: number, nextState: number[], done: boolean, logProb: number, value: number): void {
        this.experiences.push({ state, action, reward, nextState, done, logProb, value });
    }

    public update(): void {
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

        const advantages: number[] = [];
        const returns: number[] = [];

        let lastAdvantage = 0;
        for (let i = this.experiences.length - 1; i >= 0; i--) {
            const reward = rewards[i];
            const done = dones[i];
            const value = oldValues[i];
            const nextValue = i + 1 < this.experiences.length ? oldValues[i + 1] : (done ? 0 : this.critic.forward(nextStates[i])[0]);

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
                const currentActionProbabilities = miniBatchStates.map(s => this.actor.forward(s, true)); // Training mode
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
                // For simplicity, we'll assume a single output gradient for the actor loss.
                // In a real scenario, this would involve gradients w.r.t. each action probability.
                // Here, we'll use a simplified approach where the gradient is proportional to the loss.
                // This is a conceptual gradient for demonstration.
                const actorOutputGradient = currentActionProbabilities.map((probs, idx) => {
                    const actionTaken = miniBatchActions[idx];
                    const grad = Array(probs.length).fill(0);
                    // Simplified gradient: if ratio > 1+epsilon or < 1-epsilon, then gradient is non-zero
                    // This is a very rough approximation of the PPO gradient.
                    const ratio = ratios[idx];
                    const advantage = miniBatchNormalizedAdvantages[idx];

                    if (ratio > 1 + epsilon) {
                        grad[actionTaken] = -advantage; // Encourage more
                    } else if (ratio < 1 - epsilon) {
                        grad[actionTaken] = advantage; // Discourage less
                    } else {
                        grad[actionTaken] = -advantage; // Standard policy gradient
                    }
                    return grad;
                }).flat(); // Flatten the array of gradients for each action

                this.actor.zeroGrad();
                this.actor.backward(actorOutputGradient); // Pass the conceptual gradient
                this.actor.applyGradients(this.config.learningRateActor, 0.9, 0.999, 1e-8, this.config.l2RegularizationRate, this.config.clipNorm);

                // --- Critic Update ---
                // Calculate current value estimates
                const currentValues = miniBatchStates.map(s => this.critic.forward(s, true)[0]); // Training mode

                // Calculate critic loss (MSE)
                let criticLoss = 0;
                for (let j = 0; j < currentValues.length; j++) {
                    criticLoss += (currentValues[j] - miniBatchReturns[j]) ** 2;
                }
                criticLoss /= currentValues.length; // Average loss over mini-batch

                // Backpropagate critic loss
                // Gradient of MSE w.r.t. prediction: 2 * (prediction - target)
                const criticOutputGradient = currentValues.map((val, idx) => 2 * (val - miniBatchReturns[idx]));

                this.critic.zeroGrad();
                this.critic.backward(criticOutputGradient); // Pass the conceptual gradient
                this.critic.applyGradients(this.config.learningRateCritic);
            }
        }

        // Clear experiences after update
        this.experiences = [];
        console.log("PPO update completed.");
    }
}
