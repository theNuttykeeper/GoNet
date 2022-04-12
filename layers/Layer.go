package layers

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/theNuttykeeper/GoNet/tensor"
)

//Interface that stores all the methods that should be present in all layers
type Layer interface {
	perform(inputTensor *tensor.Tensor) (*tensor.Tensor, error)
}

//Struct which stores the input and output sizes that can be used in a layer
type size struct {
	inputSize  int
	outputSize int
}

//Struct which stores all the information that is needed for creating a linear layer
type Linear struct {
	size
	weights *tensor.Tensor
}

//Creates a new object of the linear layer struct and returns a pointer to it
func NewLinear(inputSize, outputSize int) (*Linear, error) {
	if inputSize < 0 || outputSize < 0 {
		return &Linear{size{0, 0}, tensor.NewTensor([]float64{})}, fmt.Errorf("invalid size arguments, all sizes must be >= 0, got %d, %d", inputSize, outputSize)
	}

	//Do not need to check error here as already confirmed both input and output sizes are greater than 0
	weightsTensor, _ := createRandomWeightsTensor(inputSize * outputSize)
	return &Linear{size{inputSize: inputSize, outputSize: outputSize}, weightsTensor}, nil
}

//Creates a tensor containing a random list of weight values (all between 0 and 1)
func createRandomWeightsTensor(length int) (*tensor.Tensor, error) {
	if length < 0 {
		return tensor.NewTensor([]float64{}), fmt.Errorf("invalid length argument, length must be >= 0, got %d", length)
	}

	rand.Seed(time.Now().UnixNano())
	weights := make([]float64, length)
	for i := range weights {
		weights[i] = rand.Float64()
	}
	return tensor.NewTensor(weights), nil
}

//Passes an input through the linear layer and returns the output from that layer
func (linear *Linear) Perform(inputTensor *tensor.Tensor) (*tensor.Tensor, error) {
	if len(inputTensor.GetData()) != linear.inputSize {
		return tensor.NewTensor([]float64{}), fmt.Errorf("input tensor not correct size for layer, expected %d, got %d", linear.inputSize, len(inputTensor.GetData()))
	}

	outputData := make([]float64, linear.outputSize)
	inputData := inputTensor.GetData()
	weightData := linear.weights.GetData()
	for i := 0; i < linear.outputSize; i++ {
		for j := 0; j < linear.inputSize; j++ {
			outputData[i] += inputData[j] * weightData[(i*linear.outputSize)+j]
		}
	}

	return tensor.NewTensor(outputData), nil
}
