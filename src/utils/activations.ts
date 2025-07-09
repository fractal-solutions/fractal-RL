// src/utils/activations.ts

export type ActivationFunction = (x: number) => number;
export type ActivationDerivative = (x: number) => number;

export const activations = {
    relu: {
        func: (x: number) => Math.max(0, x),
        deriv: (x: number) => (x > 0 ? 1 : 0)
    },
    leakyRelu: {
        func: (x: number, alpha: number = 0.01) => (x > 0 ? x : x * alpha),
        deriv: (x: number, alpha: number = 0.01) => (x > 0 ? 1 : alpha)
    },
    sigmoid: {
        func: (x: number) => 1 / (1 + Math.exp(-x)),
        deriv: (x: number) => {
            const s = 1 / (1 + Math.exp(-x));
            return s * (1 - s);
        }
    },
    tanh: {
        func: (x: number) => Math.tanh(x),
        deriv: (x: number) => 1 - Math.pow(Math.tanh(x), 2)
    },
    linear: { // For output layers where no activation is desired
        func: (x: number) => x,
        deriv: (x: number) => 1
    }
};

export function getActivation(name: string): { func: ActivationFunction, deriv: ActivationDerivative } {
    switch (name) {
        case 'relu':
            return activations.relu;
        case 'leakyRelu':
            return activations.leakyRelu;
        case 'sigmoid':
            return activations.sigmoid;
        case 'tanh':
            return activations.tanh;
        case 'linear':
            return activations.linear;
        default:
            throw new Error(`Unknown activation function: ${name}`);
    }
}
