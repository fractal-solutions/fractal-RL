// tests/dqn.test.ts
import { DQNAgent } from '../src/dqn';
import { expect, test, describe } from "bun:test";

// A very basic mock environment for testing purposes
class MockEnvironment {
    private state: number[];
    private done: boolean;
    private steps: number;
    private maxSteps: number;

    constructor(stateDim: number, maxSteps: number = 100) {
        this.state = Array(stateDim).fill(0).map(() => Math.random());
        this.done = false;
        this.steps = 0;
        this.maxSteps = maxSteps;
    }

    public reset(): number[] {
        this.state = Array(this.state.length).fill(0).map(() => Math.random());
        this.done = false;
        this.steps = 0;
        return this.state;
    }

    public step(action: number): { state: number[], reward: number, done: boolean } {
        this.steps++;
        // Simulate a state change
        this.state = this.state.map(s => s + (Math.random() - 0.5) * 0.1);
        let reward = 0.1; // Small positive reward for staying alive
        if (action === 0) reward += 0.01; // Arbitrary reward for action 0

        if (this.steps >= this.maxSteps) {
            this.done = true;
            reward = -1; // Penalty for ending
        }

        return { state: this.state, reward, done: this.done };
    }

    public isDone(): boolean {
        return this.done;
    }
}

describe('DQNAgent', () => {
    const stateDim = 4; // Example: cartpole state
    const actionDim = 2; // Example: cartpole actions (left/right)

    const config = {
        stateDim,
        actionDim,
        learningRate: 0.001,
        gamma: 0.99,
        epsilonStart: 1.0,
        epsilonEnd: 0.01,
        epsilonDecay: 0.995,
        batchSize: 32,
        targetUpdateFrequency: 10,
        replayBufferSize: 1000,
    };

    test('should initialize DQNAgent without errors', () => {
        const agent = new DQNAgent(config);
        expect(agent).toBeInstanceOf(DQNAgent);
    });

    test('selectAction should return a valid action', () => {
        const agent = new DQNAgent(config);
        const state = [0.1, 0.2, 0.3, 0.4];
        const action = agent.selectAction(state);
        expect(action).toBeGreaterThanOrEqual(0);
        expect(action).toBeLessThan(actionDim);
    });

    test('remember method should store experiences in replay buffer', () => {
        const agent = new DQNAgent(config);
        const state = [0.1, 0.2, 0.3, 0.4];
        const action = 0;
        const reward = 1;
        const nextState = [0.11, 0.21, 0.31, 0.41];
        const done = false;

        // @ts-ignore - Accessing private property for testing
        const initialBufferSize = agent.replayBuffer.size();
        agent.remember(state, action, reward, nextState, done);
        // @ts-ignore
        expect(agent.replayBuffer.size()).toBe(initialBufferSize + 1);
    });

    test('learn method should not throw error when called with enough experiences', () => {
        const agent = new DQNAgent(config);
        const env = new MockEnvironment(stateDim, config.batchSize + 5); // Ensure enough experiences

        let state = env.reset();
        let done = false;
        for (let i = 0; i < config.batchSize + 5; i++) {
            const action = agent.selectAction(state);
            const { state: nextState, reward, done: envDone } = env.step(action);
            agent.remember(state, action, reward, nextState, envDone);
            state = nextState;
            done = envDone;
            if (done) break;
        }

        // This test primarily checks if the method can be called without immediate errors.
        expect(() => agent.learn()).not.toThrow();
    });

    test('epsilon should decay after learning', () => {
        const agent = new DQNAgent(config);
        const initialEpsilon = agent['epsilon']; // Access private property

        // Fill buffer to enable learning
        for (let i = 0; i < config.batchSize; i++) {
            agent.remember([0,0,0,0], 0, 0, [0,0,0,0], false);
        }

        agent.learn();
        expect(agent['epsilon']).toBeLessThan(initialEpsilon);
        expect(agent['epsilon']).toBeGreaterThanOrEqual(config.epsilonEnd);
    });
});