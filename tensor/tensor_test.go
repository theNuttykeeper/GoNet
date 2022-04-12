package tensor

import (
	"testing"

	"github.com/theNuttykeeper/GoNet/utils"
)

//Test for the add functionality of tensors with valid inputs
func TestAddValid(t *testing.T) {
	tensor1 := NewTensor([]float64{1.0, 10.0, 5.0})
	tensor2 := NewTensor([]float64{4.0, 6.0, 3.0})
	addError := tensor1.Add(tensor2)

	if !utils.FloatSlicesEqual(tensor1.GetData(), []float64{5.0, 16.0, 8.0}) {
		t.Errorf("add result was incorrect, got %v, expected %v", tensor1.GetData(), []float64{5.0, 16.0, 8.0})
	} else if addError != nil {
		t.Errorf("add result was correctly calculated, however error was returned when no error should occur")
	}
}

//Test for the add functionality of tensors when the input is not valid (tensors of different sizes)
func TestAddInvalid(t *testing.T) {
	tensor1 := NewTensor([]float64{1.0, 10.0, 5.0, 6.0})
	tensor2 := NewTensor([]float64{4.0, 6.0, 3.0})
	addError := tensor1.Add(tensor2)

	if addError == nil {
		t.Errorf("add result was incorrect, got %v, expected error to occur due to different dimensions", tensor1.GetData())
	}
}

//Test for the subtract functionality of tensors with valid inputs
func TestSubtractValid(t *testing.T) {
	tensor1 := NewTensor([]float64{4.0, 10.0, 5.0})
	tensor2 := NewTensor([]float64{4.0, 6.0, 3.0})
	subtractError := tensor1.Subtract(tensor2)

	if !utils.FloatSlicesEqual(tensor1.GetData(), []float64{0.0, 4.0, 2.0}) {
		t.Errorf("add result was incorrect, got %v, expected %v", tensor1.GetData(), []float64{0.0, 4.0, 2.0})
	} else if subtractError != nil {
		t.Errorf("add result was correctly calculated, however error was returned when no error should occur")
	}
}

//Test for the subtract functionality of tensors when the input is not valid (tensors of different sizes)
func TestSubtractInvalid(t *testing.T) {
	tensor1 := NewTensor([]float64{1.0, 10.0, 5.0, 6.0})
	tensor2 := NewTensor([]float64{4.0, 6.0, 3.0})
	subtractError := tensor1.Subtract(tensor2)

	if subtractError == nil {
		t.Errorf("add result was incorrect, got %v, expected error to occur due to different dimensions", tensor1.GetData())
	}
}

//Test for the multiplication functionality of tensors with valid inputs
func TestMultiplicationValid(t *testing.T) {
	tensor1 := NewTensor([]float64{4.0, 10.0, 5.0})
	tensor2 := NewTensor([]float64{4.0, 6.0, 3.0})
	multiplicationError := tensor1.ElementMultiplication(tensor2)

	if !utils.FloatSlicesEqual(tensor1.GetData(), []float64{16.0, 60.0, 15.0}) {
		t.Errorf("add result was incorrect, got %v, expected %v", tensor1.GetData(), []float64{16.0, 60.0, 15.0})
	} else if multiplicationError != nil {
		t.Errorf("add result was correctly calculated, however error was returned when no error should occur")
	}
}

//Test for the multiplication functionality of tensors when the input is not valid (tensors of different sizes)
func TestMultiplicationInvalid(t *testing.T) {
	tensor1 := NewTensor([]float64{1.0, 10.0, 5.0, 6.0})
	tensor2 := NewTensor([]float64{4.0, 6.0, 3.0})
	multiplicationError := tensor1.ElementMultiplication(tensor2)

	if multiplicationError == nil {
		t.Errorf("add result was incorrect, got %v, expected error to occur due to different dimensions", tensor1.GetData())
	}
}

//Tests the dot product functionality for tensors when the input is valid
func TestDotProductValid(t *testing.T) {
	tensor1 := NewTensor([]float64{1.0, 10.0, 5.0})
	tensor2 := NewTensor([]float64{4.0, 6.0, 3.0})
	dotResult, dotError := tensor1.DotProduct(tensor2)

	if dotResult != 79.0 || dotError != nil {
		t.Errorf("dot product result was incorrect, got %f, expected %f", dotResult, 79.0)
	} else if dotError != nil {
		t.Errorf("Dot product result was correctly calculated, however error was returned when no error should occur")
	}
}

//Tests the dot product functionality for tensors when the input is invalid (tensors of different sizes)
func TestDotProductInvalid(t *testing.T) {
	tensor1 := NewTensor([]float64{1.0, 10.0, 5.0, 6.0})
	tensor2 := NewTensor([]float64{4.0, 6.0, 3.0})
	dotResult, dotError := tensor1.DotProduct(tensor2)

	if dotError == nil {
		t.Errorf("dot product result was incorrect, got %f, expected error due to different dimensions", dotResult)
	}
}
