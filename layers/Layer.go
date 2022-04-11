package layers

import "fmt"

type size struct {
	inputSize  int32
	outputSize int32
}

type Linear struct {
	size
}

func (linear *Linear) Details() string {
	return fmt.Sprintf("Input Size: %d, Output Size: %d", linear.inputSize, linear.outputSize)
}

func NewLinear(inputSize, outputSize int32) *Linear {
	return &Linear{size{inputSize: inputSize, outputSize: outputSize}}
}
