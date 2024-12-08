using System;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        private int[] _structure; // Содержит количество нейронов в каждом слое
        private double[][] _neurons; // Значения нейронов
        private double[][][] _weights; // Весовые коэффициенты между слоями
        private double[][] _biases; // Смещения нейронов
        private Random _random;

        public StudentNetwork(int[] structure)
        {
            _structure = structure;
            _neurons = new double[structure.Length][];
            _weights = new double[structure.Length - 1][][];
            _biases = new double[structure.Length - 1][];
            _random = new Random();

            // Инициализация нейронов
            for (int i = 0; i < structure.Length; i++)
                _neurons[i] = new double[structure[i]];

            // Инициализация весов и смещений
            for (int i = 0; i < structure.Length - 1; i++)
            {
                int currentLayerSize = structure[i];
                int nextLayerSize = structure[i + 1];

                _weights[i] = new double[nextLayerSize][];
                _biases[i] = new double[nextLayerSize];

                for (int j = 0; j < nextLayerSize; j++)
                {
                    _weights[i][j] = new double[currentLayerSize];
                    for (int k = 0; k < currentLayerSize; k++)
                        _weights[i][j][k] = _random.NextDouble() * 2 - 1; // Случайные веса
                    _biases[i][j] = _random.NextDouble() * 2 - 1; // Случайные смещения
                }
            }
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            double error = int.MaxValue;
            int iters = 0;

            while (error > acceptableError - 0.001)
            {
                // Обратное распространение ошибки
                error = Backpropagate(sample.input, sample.Output);
                iters++;
            }

            return iters;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            double totalError = 0;

            for (int epoch = 0; epoch < epochsCount; epoch++)
            {
                totalError = 0;

                foreach (Sample sample in samplesSet)
                    totalError += Backpropagate(sample.input, sample.Output);

                totalError /= samplesSet.Count;

                if (totalError <= acceptableError)
                    break;
            }

            return totalError;
        }

        protected override double[] Compute(double[] input)
        {
            Array.Copy(input, _neurons[0], input.Length);

            for (int layer = 0; layer < _weights.Length; layer++)
            {
                for (int neuron = 0; neuron < _neurons[layer + 1].Length; neuron++)
                {
                    double sum = _biases[layer][neuron];

                    for (int prevNeuron = 0; prevNeuron < _neurons[layer].Length; prevNeuron++)
                        sum += _neurons[layer][prevNeuron] * _weights[layer][neuron][prevNeuron];
                    _neurons[layer + 1][neuron] = Sigmoid(sum);
                }
            }

            return _neurons[_neurons.Length - 1];
        }

        private double Backpropagate(double[] inputs, double[] expectedOutputs)
        {
            // Прямой проход
            var outputs = Compute(inputs);

            // Вычисление ошибки
            double error = 0;
            double[][] errors = new double[_structure.Length][];
            for (int i = 0; i < _structure.Length; i++)
                errors[i] = new double[_structure[i]];

            for (int i = 0; i < expectedOutputs.Length; i++)
            {
                errors[errors.Length - 1][i] = outputs[i] - expectedOutputs[i];
                error += Math.Pow(errors[errors.Length - 1][i], 2);
            }
            error /= 2;

            // Обратное распространение
            for (int layer = _weights.Length - 1; layer >= 0; layer--)
            {
                for (int neuron = 0; neuron < _weights[layer].Length; neuron++)
                {
                    double delta = errors[layer + 1][neuron] * SigmoidDerivative(_neurons[layer + 1][neuron]);

                    for (int prevNeuron = 0; prevNeuron < _neurons[layer].Length; prevNeuron++)
                    {
                        errors[layer][prevNeuron] += delta * _weights[layer][neuron][prevNeuron];
                        _weights[layer][neuron][prevNeuron] -= delta * _neurons[layer][prevNeuron];
                    }

                    _biases[layer][neuron] -= delta;
                }
            }

            return error;
        }

        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        private double SigmoidDerivative(double x)
        {
            return x * (1 - x);
        }
    }
}
