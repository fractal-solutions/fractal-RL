# Fractal-RL

Fractal-RL is a lightweight and easy-to-use reinforcement learning library for TypeScript and JavaScript. It provides implementations of popular deep reinforcement learning algorithms, including Deep Q-Network (DQN) and Proximal Policy Optimization (PPO).

## Installation

To install the library, use npm:

```bash
npm install fractal-rl
```

## Deep Q-Network (DQN) Agent

The DQN agent is a value-based reinforcement learning algorithm that is well-suited for environments with discrete action spaces. The implementation in Fractal-RL is designed to be easy to use and configure.

### DQN Usage Example

Here is a basic example of how to use the `DQNAgent` to solve the CartPole environment:

```javascript
import { DQNAgent } from 'fractal-rl';

// Define the environment
class CartPoleEnvironment {
    // ... (environment implementation)
}

// Configure the DQN agent
const config = {
    stateDim: 4,
    actionDim: 2,
    learningRate: 0.01,
    gamma: 0.99,
    epsilonStart: 1.0,
    epsilonEnd: 0.01,
    epsilonDecay: 0.9995,
    batchSize: 64,
    targetUpdateFrequency: 50,
    replayBufferSize: 10000,
    dropoutRate: 0.0,
    l2RegularizationRate: 0.0001,
    clipNorm: 1.0,
    hiddenActivation: 'leakyRelu',
};

const agent = new DQNAgent(config);
const env = new CartPoleEnvironment();

// Training loop
for (let episode = 0; episode < 5000; episode++) {
    let state = env.reset();
    let done = false;
    while (!done) {
        const action = agent.selectAction(state);
        const { state: nextState, reward, done: envDone } = env.step(action);
        agent.remember(state, action, reward, nextState, envDone);
        if (agent.replayBuffer.size() > config.batchSize) {
            agent.learn();
        }
        state = nextState;
        done = envDone;
    }
}
```

### DQN Hyperparameters

-   `stateDim`: The dimension of the state space.
-   `actionDim`: The dimension of the action space.
-   `learningRate`: The learning rate for the optimizer.
-   `gamma`: The discount factor for future rewards.
-   `epsilonStart`: The initial value of epsilon for the epsilon-greedy exploration strategy.
-   `epsilonEnd`: The final value of epsilon.
-   `epsilonDecay`: The decay rate of epsilon.
-   `batchSize`: The size of the mini-batch for training.
-   `targetUpdateFrequency`: The frequency (in steps) at which the target network is updated.
-   `replayBufferSize`: The maximum size of the replay buffer.
-   `dropoutRate`: The dropout rate for regularization.
-   `l2RegularizationRate`: The L2 regularization rate.
-   `clipNorm`: The maximum norm for gradient clipping.
-   `hiddenActivation`: The activation function for the hidden layers.

## Proximal Policy Optimization (PPO) Agent

The PPO agent is a policy-based reinforcement learning algorithm that is well-suited for environments with continuous or discrete action spaces. The implementation in Fractal-RL is designed to be easy to use and configure.

### PPO Usage Example

Here is a basic example of how to use the `PPOAgent` to solve the CartPole environment:

```javascript
import { PPOAgent } from 'fractal-rl';

// Define the environment
class CartPoleEnvironment {
    // ... (environment implementation)
}

// Configure the PPO agent
const config = {
    stateDim: 4,
    actionDim: 2,
    learningRateActor: 0.0003,
    learningRateCritic: 0.0001,
    gamma: 0.95,
    lambda: 0.9,
    epsilon: 0.02,
    ppoEpochs: 5,
    miniBatchSize: 64,
    dropoutRate: 0.0,
    l2RegularizationRate: 0.000001,
    clipNorm: 1.0,
    hiddenActivation: 'leakyRelu',
};

const agent = new PPOAgent(config);
const env = new CartPoleEnvironment();

// Training loop
for (let episode = 0; episode < 5000; episode++) {
    let state = env.reset();
    let done = false;
    while (!done) {
        const { action, logProb, value } = agent.selectAction(state);
        const { state: nextState, reward, done: envDone } = env.step(action);
        agent.remember(state, action, reward, nextState, envDone, logProb, value);
        state = nextState;
        done = envDone;
    }
    agent.update();
}
```

### PPO Hyperparameters

-   `stateDim`: The dimension of the state space.
-   `actionDim`: The dimension of the action space.
-   `learningRateActor`: The learning rate for the actor network.
-   `learningRateCritic`: The learning rate for the critic network.
-   `gamma`: The discount factor for future rewards.
-   `lambda`: The lambda parameter for Generalized Advantage Estimation (GAE).
-   `epsilon`: The clipping parameter for the PPO objective function.
-   `ppoEpochs`: The number of epochs to run when updating the policy.
-   `miniBatchSize`: The size of the mini-batch for training.
-   `dropoutRate`: The dropout rate for regularization.
-   `l2RegularizationRate`: The L2 regularization rate.
-   `clipNorm`: The maximum norm for gradient clipping.
-   `hiddenActivation`: The activation function for the hidden layers.

## PPO CartPole GUI

To run the PPO CartPole GUI, start the development server:

```bash
bun run start
```

Then, open your web browser and navigate to `http://localhost:4000/index.html`.
