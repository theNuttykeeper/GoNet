package layers

import (
	"math"

	"github.com/theNuttykeeper/GoNet/tensor"
)

//Empty ReLU struct used so that we can implement Layer interface
type ReLU struct{}

//Empty Sigmoid struct used so that we can implement Layer interface
type Sigmoid struct{}

//Layer interface method that defines the action of the ReLU activation on an input
func (relu *ReLU) Perform(inputTensor *tensor.Tensor) (*tensor.Tensor, error) {
	outputData := make([]float64, len(inputTensor.GetData()))

	for i, element := range inputTensor.GetData() {
		outputData[i] = math.Max(0.0, element)
	}

	return tensor.NewTensor(outputData), nil
}

//Layer interface method that defines the action of the Sigmoid activation on an input
func (sigmoid *Sigmoid) Perform(inputTensor *tensor.Tensor) (*tensor.Tensor, error) {
	outputData := make([]float64, len(inputTensor.GetData()))

	for i, element := range inputTensor.GetData() {
		outputData[i] = 1 / (1 + math.Exp(-element))
	}

	return tensor.NewTensor(outputData), nil
}
