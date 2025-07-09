
// tests/ppo.test.ts
import { PPOAgent } from '../src/ppo';
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

describe('PPOAgent', () => {
    const stateDim = 4; // Example: cartpole state
    const actionDim = 2; // Example: cartpole actions (left/right)

    const config = {
        stateDim,
        actionDim,
        learningRateActor: 0.0003,
        learningRateCritic: 0.001,
        gamma: 0.99,
        lambda: 0.95,
        epsilon: 0.2,
        ppoEpochs: 3, // Reduced for faster test
        miniBatchSize: 64,
    };

    test('should initialize PPOAgent without errors', () => {
        const agent = new PPOAgent(config);
        expect(agent).toBeInstanceOf(PPOAgent);
    });

    test('selectAction should return a valid action, logProb, and value', () => {
        const agent = new PPOAgent(config);
        const state = [0.1, 0.2, 0.3, 0.4];
        const { action, logProb, value } = agent.selectAction(state);
        expect(action).toBeGreaterThanOrEqual(0);
        expect(action).toBeLessThan(actionDim);
        expect(typeof logProb).toBe('number');
        expect(typeof value).toBe('number');
    });

    test('remember method should store experiences', () => {
        const agent = new PPOAgent(config);
        const state = [0.1, 0.2, 0.3, 0.4];
        const action = 0;
        const reward = 1;
        const nextState = [0.11, 0.21, 0.31, 0.41];
        const done = false;
        const logProb = -0.5;
        const value = 0.8;

        // @ts-ignore - Accessing private property for testing
        const initialExperiencesLength = agent.experiences.length;
        agent.remember(state, action, reward, nextState, done, logProb, value);
        // @ts-ignore
        expect(agent.experiences.length).toBe(initialExperiencesLength + 1);
    });

    test('update method should process experiences and clear buffer', () => {
        const agent = new PPOAgent(config);
        const env = new MockEnvironment(stateDim, 5); // Short episode for test

        let state = env.reset();
        let done = false;
        while (!done) {
            const { action, logProb, value } = agent.selectAction(state);
            const { state: nextState, reward, done: envDone } = env.step(action);
            agent.remember(state, action, reward, nextState, envDone, logProb, value);
            state = nextState;
            done = envDone;
        }

        // @ts-ignore
        expect(agent.experiences.length).toBeGreaterThan(0);
        expect(() => agent.update()).not.toThrow();
        // @ts-ignore
        expect(agent.experiences.length).toBe(0); // Buffer should be cleared after update
    });

    
});
