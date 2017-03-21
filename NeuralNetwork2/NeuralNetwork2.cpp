// NeuralNetwork2.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <assert.h>
#include <cmath>
using namespace std;

/*This is a structure containing the data for connections between
Neurons */
struct Connection
{
	double weight; //The weight to the subsequent Neuron
double deltaWeight; //Value used in training to update this weight

};


class Neuron; // Declaration of Class Neuron for typdef Layer
typedef vector<Neuron> Layer; //A Layer is a Vector of Neurons

//~~~~~~~~~~~~~~~~~~~~~~~~~ Class Neuron ~~~~~~~~~~~~~~~~~~~
class Neuron {
public:
	Neuron(unsigned numOutputs, unsigned myIndex); //Constructor for Class Neuron
	void feedForward(const Layer &previousLayer); //Perform forward pass on inputs
	void setOutputValue(double value) { m_outputVal = value; };//set the outputvalue for our neuron
	double getOutputValue()const { return m_outputVal; };//set the outputvalue for our neuron

private:
	double m_myindex;
	double m_outputVal;  //The output value of the neuron
	vector<Connection> m_outputWeights; //Output Weight and little Delta Weight
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	static double randomWeight() {
		return rand() / double(RAND_MAX);
	}
};

double Neuron::transferFunction(double x) {
	//Hyperbolic tangent -1 to 1
	return (double)tan(x);
}

double Neuron::transferFunctionDerivative(double x) {
	//Return the derivative of hyperbolic tangent
	return 1.0 - x*x;
}

void Neuron::feedForward(const Layer &previousLayer) {
	double sum = 0.0;
	for (int n = 0; n < previousLayer.size(); n++) {
		double prevOutput = previousLayer[n].getOutputValue();
		double prevWeight = previousLayer[n].m_outputWeights[m_myindex].weight;
		sum += prevOutput*prevWeight;
	}
	m_outputVal = Neuron::transferFunction(sum);
}
/*Class Neuron constructor takes the number of output weights on this neuron*/
Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {
	m_myindex = myIndex;
	//For each specified output, create a new connection and append to m_outputWeights
	for (int c = 0; c < numOutputs; c++) {
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}
}

//~~~~~~~~~~~~~~~~~~~~~~~~~ Class Neuron ~~~~~~~~~~~~~~~~~~~

/*The NeuralNetwork Class is the manager of Layers and Neurons*/
class Net
{
public:
	//Constructor takes topology which is a list of Layer Lengths
	Net(const vector<unsigned> &topology);

	//Passes input values forward through the network
	void feedForward(const vector<double> &inputVals);

	//Train the network using desired, or targetVals
	void backProp(const vector<double> &targetVals);

	//Puts the Results or Estimations in resultVals
	void getResults(vector<double> &resultVals);

private:
	//Remeber Layers are Vectors themselves. This is 2D
	vector<Layer> m_layers; //m_layers [layerNum] [neuronNum]
	double m_error;
	double m_recentAvgError;
	double m_recentAvgSmoothingFactor;
};

void Net::backProp(const vector<double> &targetVals) {
	//Calculate overall net error (RMS of output neuron errors)
	Layer &outputLayer = m_layers.back();
	m_error = 0.0;
	for (int outputNeuron = 0; outputNeuron < outputLayer.size() - 1; outputNeuron++) {
		double yieldedValue = outputLayer[outputNeuron].getOutputValue();
		double target = targetVals[outputNeuron];
		double delta = target - yieldedValue;
		m_error += delta * delta;
	}
	m_error /= outputLayer.size() - 1; // Average overall error
	m_error = sqrt(m_error); //RMS

	//Implement a recent avg error measurement
	m_recentAvgError = (m_recentAvgError * m_recentAvgSmoothingFactor + m_error)
		/ (m_recentAvgSmoothingFactor + 1.0);

	//Calculate output layer gradients
	for (int n = 0; n < outputLayer.size()-1; n++){
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	//Calculate Hidden layer Gradients
	for (int layerNum = m_layers.size()-2; layerNum >0; layerNum++) {
		Layer & hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];

		//For each neuron in the layer, calculate its gradient
		for (int n = 0; n < hiddenLayer.size(); n++) {
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	/*For all layers from outputs to first hidden layer
	Update the weight connections */
}

void Net::feedForward(const vector<double> &inputVals) {
	assert(inputVals.size() == m_layers[0].size() - 1);

	for (int input = 0; input < inputVals.size(); input++) {
		double nextValue = inputVals[input];
		m_layers[0][input].setOutputValue(nextValue);
	}

	for (int layer = 1; layer < m_layers.size(); layer++) {
		//Get a reference to the previous layer for computation of current layer
		Layer &previousLayer = m_layers[layer - 1];
		for (int neuron = 0; neuron < m_layers[layer].size(); neuron++) {
			//Neurons have their own feedForward Function as well
			m_layers[layer][neuron].feedForward(previousLayer);
		}
	}
}

Net::Net(const vector<unsigned> &topology) {
	//The size is the # of layers where the value at each index is the # Neurons
	unsigned numLayers = topology.size();
	for (int layerNum = 0; layerNum < numLayers; ++layerNum) {
		m_layers.push_back(Layer());

		//If the layerNum is the last, it has 0 weights or Connections
		unsigned numOfOutputs = layerNum == topology.size()-1 ? 0 : topology[layerNum + 1];

		/*Get the number of Neurons needed for our layerNum and create
		new Neurons. Append those to m_layers */
		for (int neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
			m_layers.back().push_back(Neuron(numOfOutputs, neuronNum));
			cout << "Neuron: "<<neuronNum<< " added to Layer: "<< layerNum << endl;
		}
	}
}

int main()
{	
	vector<unsigned> topology;
	topology.push_back(3);
	topology.push_back(2);
	topology.push_back(1);

	Net myNet(topology);

	vector<double> inputVals;
	myNet.feedForward(inputVals);

	vector<double> targetVals;
	myNet.backProp(targetVals);

	vector<double> resultVals;
	myNet.getResults(resultVals);

	system("pause");
    return 0;
}

