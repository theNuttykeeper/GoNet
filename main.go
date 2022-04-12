package main

import (
	"fmt"

	"github.com/theNuttykeeper/GoNet/layers"
	"github.com/theNuttykeeper/GoNet/tensor"
)

func main() {
	fmt.Println("This is the GoNet Neural Network Package")

	linear, linearError := layers.NewLinear(10, 5)
	if linearError == nil {
		inputTensor := tensor.NewTensor([]float64{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5})

		outputData, outputError := linear.Perform(inputTensor)
		if outputError == nil {
			fmt.Printf("Output from linear layer: %v\n", outputData)
		} else {
			fmt.Printf("Error Occurred: %v\n", outputError)
		}
	}
}
