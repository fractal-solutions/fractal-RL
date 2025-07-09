# fractal-rl

To install dependencies:

```bash
bun install
```

To run:

```bash
bun run index.ts
```

This project was created using `bun init` in bun v1.2.18. [Bun](https://bun.sh) is a fast all-in-one JavaScript runtime.

## Deep Q-Network (DQN) Agent

The DQN agent is implemented in `src/dqn/`. You can run a training example using `dqn-test.js`.

### How to Run DQN Example

```bash
bun run dqn-test.js
```

### DQN Configuration Parameters

The `config` object in `dqn-test.js` defines the parameters for the DQN agent:

*   `stateDim`: Dimension of the observation space.
*   `actionDim`: Dimension of the action space.
*   `learningRate`: Learning rate for the Q-network optimizer.
*   `gamma`: Discount factor for future rewards.
*   `epsilonStart`: Initial value of epsilon for epsilon-greedy action selection.
*   `epsilonEnd`: Minimum value of epsilon.
*   `epsilonDecay`: Decay rate for epsilon.
*   `batchSize`: Number of experiences to sample from the replay buffer for learning.
*   `targetUpdateFrequency`: How often the target Q-network is updated (in terms of learning steps).
*   `replayBufferSize`: Maximum capacity of the replay buffer.
*   `dropoutRate`: Dropout rate for regularization in the Q-network (0.0 for no dropout).
*   `l2RegularizationRate`: L2 regularization rate for the Q-network weights.
*   `clipNorm`: Gradient clipping norm to prevent exploding gradients.
*   `hiddenActivation`: Activation function for the hidden layers of the Q-network. Available options: `'relu'`, `'leakyRelu'`, `'sigmoid'`, `'tanh'`, `'linear'`.

## Proximal Policy Optimization (PPO) Agent

The PPO agent is implemented in `src/ppo/`. You can run a training example using `ppo-test.js`.

### How to Run PPO Example

```bash
bun run ppo-test.js
```

### PPO Configuration Parameters

The `config` object in `ppo-test.js` defines the parameters for the PPO agent:

*   `stateDim`: Dimension of the observation space.
*   `actionDim`: Dimension of the action space.
*   `learningRateActor`: Learning rate for the actor network optimizer.
*   `learningRateCritic`: Learning rate for the critic network optimizer.
*   `gamma`: Discount factor for future rewards.
*   `lambda`: GAE (Generalized Advantage Estimation) parameter.
*   `epsilon`: Clip parameter for the PPO objective.
*   `ppoEpochs`: Number of optimization epochs per update.
*   `miniBatchSize`: Number of experiences in each mini-batch during optimization.
*   `dropoutRate`: Dropout rate for regularization in the actor and critic networks (0.0 for no dropout).
*   `l2RegularizationRate`: L2 regularization rate for the network weights.
*   `clipNorm`: Gradient clipping norm to prevent exploding gradients.
*   `hiddenActivation`: Activation function for the hidden layers of the actor and critic networks. Available options: `'relu'`, `'leakyRelu'`, `'sigmoid'`, `'tanh'`, `'linear'`.

## PPO CartPole GUI

To run the PPO CartPole GUI, start the development server:

```bash
bun run start
```

Then, open your web browser and navigate to `http://localhost:3000/ppo.html`.