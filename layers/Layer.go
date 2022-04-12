package layers

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/theNuttykeeper/GoNet/tensor"
)

type size struct {
	inputSize  int32
	outputSize int32
}

type Linear struct {
	size
	weights *tensor.Tensor
}

func (linear *Linear) Details() string {
	return fmt.Sprintf("Input Size: %d, Output Size: %d, Current Weights: %f", linear.inputSize, linear.outputSize, linear.weights.GetData())
}

func NewLinear(inputSize, outputSize int32) *Linear {
	return &Linear{size{inputSize: inputSize, outputSize: outputSize}, createRandomWeightsTensor(inputSize * outputSize)}
}

//Creates a tensor containing a random list of weight values (all between 0 and 1)
func createRandomWeightsTensor(length int32) *tensor.Tensor {
	rand.Seed(time.Now().UnixNano())

	weights := make([]float64, length)
	for i := range weights {
		weights[i] = rand.Float64()
	}
	return tensor.NewTensor(weights)
}
