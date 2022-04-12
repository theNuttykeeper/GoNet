package layers

import (
	"testing"

	"github.com/theNuttykeeper/GoNet/tensor"
)

//Tests that the create random weights function correctly generates a slice of the right length containing values between 0 and 1.
//This test uses valid length input
func TestCreateRandomWeightsValid(t *testing.T) {
	var length int = 20
	weights, weightsError := createRandomWeightsTensor(length)

	if len(weights.GetData()) != length {
		t.Errorf("weight list not initialized to correct length, expected length: %d, got %d", length, len(weights.GetData()))
	} else if weightsError != nil {
		t.Errorf("weight list initalized to correct length but error returned: %v", weightsError)
	}

	for _, element := range weights.GetData() {
		if element < 0.0 || element > 1.0 {
			t.Errorf("value in weight list outside expected range, expected range 0.0 - 1.0, got %f", element)
		}
	}
}

//Tests that the create random weights function correctly returns error when a negative length value is provided
func TestCreateRandomWeightsInvalid(t *testing.T) {
	var length int = -1
	_, weightsError := createRandomWeightsTensor(length)

	if weightsError == nil {
		t.Errorf("weights error was expected when initialized with negative length, however no error occurred")
	}
}

//Tests that we can correct create a new linear layer using the NewLinear function - using valid size inputs
func TestNewLinearValid(t *testing.T) {
	layer, layerError := NewLinear(1, 1)

	if layerError != nil {
		t.Errorf("error when creating new linear layer, error: %v", layerError)
	} else if len(layer.weights.GetData()) != 1 {
		t.Errorf("linear layer not created to correct size, expected size 1, got %d", len(layer.weights.GetData()))
	}
}

//Tests that the linear layer creation provides an error when invalid sizes are given
func TestNewLinearInvalid(t *testing.T) {
	_, layer1Error := NewLinear(-1, 1)
	_, layer2Error := NewLinear(1, -1)
	_, layer3Error := NewLinear(-1, -1)

	if layer1Error == nil {
		t.Errorf("expected error when creating layer with negative input size")
	}
	if layer2Error == nil {
		t.Errorf("expected error when creating layer with negative output size")
	}
	if layer3Error == nil {
		t.Errorf("expected error when creating lauer with negative input and output size")
	}
}

//Tests that the perform function works correctly when input to the layer is valid
func TestPerformValid(t *testing.T) {
	layer, layerError := NewLinear(10, 5)

	if layerError != nil {
		t.Errorf("error when creating the linear layer: error: %v", layerError)
	}

	inputTensor := tensor.NewTensor([]float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0})
	outputTensor, outputError := layer.Perform(inputTensor)

	if outputError != nil {
		t.Errorf("error when getting the output of the layer, error: %v", outputError)
	} else if len(outputTensor.GetData()) != 5 {
		t.Errorf("size of output tensor not correct, expected 5, got %d", len(outputTensor.GetData()))
	}
}

//Tests that the perform function gives error when the input tensor is not of the correct size
func TestPerformInvalid(t *testing.T) {
	layer, layerError := NewLinear(10, 5)

	if layerError != nil {
		t.Errorf("error when creating the linear layer: error: %v", layerError)
	}

	inputTensor := tensor.NewTensor([]float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0})
	_, outputError := layer.Perform(inputTensor)

	if outputError == nil {
		t.Errorf("expected error when passing input to layer with incorrect size")
	}
}
