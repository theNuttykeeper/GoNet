package layers

import (
	"math"
	"testing"

	"github.com/theNuttykeeper/GoNet/tensor"
	"github.com/theNuttykeeper/GoNet/utils"
)

//Tests the perform operation of the relu activation
func TestReLUPerform(t *testing.T) {
	relu := ReLU{}
	inputTensor := tensor.NewTensor([]float64{1.0, 0.0, -1.0})

	//Error will alwauys be nil from relu perform
	outputTensor, _ := relu.Perform(inputTensor)
	expectedOutput := []float64{1.0, 0.0, 0.0}

	if !utils.FloatSlicesEqual(outputTensor.GetData(), expectedOutput) {
		t.Errorf("Output of ReLU activation not correct, got %v, expected %v", outputTensor.GetData(), expectedOutput)
	}
}

//Tests the perform operation of the sigmoid activation
func TestSigmoidPerform(t *testing.T) {
	sigmoid := Sigmoid{}
	inputTensor := tensor.NewTensor([]float64{0.0, 10, -10, 1, -1})

	outputTensor, _ := sigmoid.Perform(inputTensor)
	expectedOutput := []float64{0.5, 1 / (1 + math.Exp(-10)), 1 / (1 + math.Exp(10)), 1 / (1 + math.Exp(-1)), 1 / (1 + math.Exp(1))}

	if !utils.FloatSlicesEqual(outputTensor.GetData(), expectedOutput) {
		t.Errorf("Output of Sigmoid activation not correct, got %v, expected %v", outputTensor.GetData(), expectedOutput)
	}
}
