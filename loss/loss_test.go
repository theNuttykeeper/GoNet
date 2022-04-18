package loss

import (
	"testing"

	"github.com/theNuttykeeper/GoNet/tensor"
)

//Test for the MSE loss with valid input
func TestMSELossValid(t *testing.T) {
	actualOutputTensor := tensor.NewTensor([]float64{1.0, 2.0, 3.0})
	predictedOutputTensor := tensor.NewTensor([]float64{0.0, 3.0, 2.0})
	mseResult, mseError := MSELoss(predictedOutputTensor, actualOutputTensor)
	expectedResult := 3.0

	if mseResult != expectedResult {
		t.Errorf("MSE loss function did not give expected resulted, expected %f but got expected %f", expectedResult, mseResult)
	}
	if mseError != nil {
		t.Errorf("MSE Loss function gave unexpected error %s", mseError)
	}
}

//Test for the MSE loss with invalid input
func TestMSELossInvalid(t *testing.T) {
	actualOutputTensor := tensor.NewTensor([]float64{1.0, 2.0, 3.0})
	predictedOutputTensor := tensor.NewTensor([]float64{1.0, 2.0, 3.0, 4.0})
	mseResult, mseError := MSELoss(predictedOutputTensor, actualOutputTensor)

	if mseError == nil {
		t.Errorf("MSE result with invalid input incorrect, expected error but got %f", mseResult)
	}
}
