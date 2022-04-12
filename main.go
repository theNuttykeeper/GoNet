package main

import (
	"fmt"

	"github.com/theNuttykeeper/GoNet/layers"
	"github.com/theNuttykeeper/GoNet/tensor"
)

func main() {
	fmt.Println("This is the GoNet Neural Network Package")

	linear, linearError := layers.NewLinear(10, 5)
	relu := layers.Sigmoid{}
	if linearError == nil {
		inputTensor := tensor.NewTensor([]float64{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5})

		outputData, outputError := linear.Perform(inputTensor)
		outputData, _ = relu.Perform(outputData)
		if outputError == nil {
			fmt.Printf("Output from linear layer followed by relu: %v\n", outputData)
		} else {
			fmt.Printf("Error Occurred: %v\n", outputError)
		}
	}
}
