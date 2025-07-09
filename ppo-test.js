// ppo-test.js
import { PPOAgent } from './src/ppo/index.js'; // Note: .js extension for runtime

// A simple CartPole-like environment for demonstration
class CartPoleEnvironment {
    constructor() {
        this.state = [0.0, 0.0, 0.0, 0.0]; // cart position, cart velocity, pole angle, pole velocity
        this.steps = 0;
        this.maxSteps = 200;
        this.poleAngleThreshold = 0.2095; // ~12 degrees
        this.cartPositionThreshold = 2.4;
    }

    reset() {
        this.state = Array(4).fill(0).map(() => (Math.random() - 0.5) * 0.1); // Small random initial state
        this.steps = 0;
        return this.state;
    }

    step(action) {
        // Simplified physics for demonstration
        const x = this.state[0];
        const x_dot = this.state[1];
        const theta = this.state[2];
        const theta_dot = this.state[3];

        const force = (action === 1) ? 10.0 : -10.0; // Apply force left or right

        const costheta = Math.cos(theta);
        const sintheta = Math.sin(theta);

        const temp = (force + 0.05 * theta_dot * theta_dot * sintheta) / 1.0; // Simplified mass_pole + mass_cart = 1.0

        const thetaacc = (9.8 * sintheta - costheta * temp) / (0.5 * (4/3 - 0.05 * costheta * costheta)); // Simplified length_pole = 0.5, mass_pole = 0.05
        const xacc = temp - 0.05 * thetaacc * costheta;

        // Update state (Euler integration)
        this.state[0] = x + 0.02 * x_dot;
        this.state[1] = x_dot + 0.02 * xacc;
        this.state[2] = theta + 0.02 * theta_dot;
        this.state[3] = theta_dot + 0.02 * thetaacc;

        this.steps++;

        const done =
            this.state[0] < -this.cartPositionThreshold ||
            this.state[0] > this.cartPositionThreshold ||
            this.state[2] < -this.poleAngleThreshold ||
            this.state[2] > this.poleAngleThreshold ||
            this.steps >= this.maxSteps;

        const reward = done ? 0 : 1; // Reward 1 for each step pole is upright

        return { state: this.state, reward, done };
    }
}

async function runPPOAgent() {
    const stateDim = 4;
    const actionDim = 2; // Left or Right

    const config = {
        stateDim: stateDim,
        actionDim: actionDim,
        learningRateActor: 0.0003,
        learningRateCritic: 0.000001,
        gamma: 0.95,
        lambda: 0.9,
        epsilon: 0.02,
        ppoEpochs: 5,
        miniBatchSize: 64,
        dropoutRate: 0.0, // No dropout for this simple example
        l2RegularizationRate: 0.000001,
        clipNorm: 1.0,
        hiddenActivation: 'leakyRelu', // Added activation function option
    };

    const agent = new PPOAgent(config);
    const env = new CartPoleEnvironment();

    const numTrainingEpisodes = 5000; // Maximum training episodes
    const maxStepsPerEpisode = 200;
    const solvedRewardThreshold = maxStepsPerEpisode-1; // Reward to consider an episode "solved"
    const consecutiveSolvedEpisodesThreshold = 3; // Number of consecutive solved episodes to trigger evaluation
    const numEvaluationEpisodes = 100; // Number of episodes to run in evaluation mode

    let consecutiveSolvedEpisodes = 0;
    let evaluationPhase = false;
    let evaluationEpisodeCounter = 0;

    console.log("Starting PPO Agent training...");

    for (let episode = 0; episode < numTrainingEpisodes; episode++) {
        let state = env.reset();
        let done = false;
        let totalReward = 0;
        let stepsInEpisode = 0;

        while (!done && stepsInEpisode < maxStepsPerEpisode) {
            const { action, logProb, value } = agent.selectAction(state);
            const { state: nextState, reward, done: envDone } = env.step(action);

            if (!evaluationPhase) {
                agent.remember(state, action, reward, nextState, envDone, logProb, value);
            }

            state = nextState;
            totalReward += reward;
            done = envDone;
            stepsInEpisode++;
        }

        if (!evaluationPhase) {
            agent.update(); // Update PPO agent after each episode during training
        }

        console.log(`Episode ${episode + 1} (${evaluationPhase ? 'Evaluation' : 'Training'}): Total Reward = ${totalReward}, Steps = ${stepsInEpisode}`);

        if (totalReward >= solvedRewardThreshold) {
            consecutiveSolvedEpisodes++;
            if (consecutiveSolvedEpisodes >= consecutiveSolvedEpisodesThreshold && !evaluationPhase) {
                console.log(`Agent solved the environment for ${consecutiveSolvedEpisodesThreshold} consecutive episodes. Entering evaluation phase.`);
                evaluationPhase = true;
            }
        } else {
            consecutiveSolvedEpisodes = 0; // Reset counter if not solved
        }

        if (evaluationPhase) {
            evaluationEpisodeCounter++;
            if (evaluationEpisodeCounter >= numEvaluationEpisodes) {
                console.log(`Completed ${numEvaluationEpisodes} evaluation episodes.`);
                break; // End simulation after evaluation
            }
        }

        if (episode + 1 === numTrainingEpisodes && !evaluationPhase) {
            console.log("Maximum training episodes reached. Training finished.");
        }
    }
    console.log("PPO Agent simulation finished.");
}

runPPOAgent();
