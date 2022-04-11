package tensor

import (
	"testing"

	"github.com/theNuttykeeper/GoNet/utils"
)

//Test for the add functionality of tensors
func TestAdd(t *testing.T) {
	tensor1 := NewTensor([]float64{1.0, 10.0, 5.0})
	tensor2 := NewTensor([]float64{4.0, 6.0, 3.0})
	tensor1.Add(tensor2)

	if !utils.FloatSlicesEqual(tensor1.GetData(), []float64{5.0, 16.0, 8.0}) {
		t.Errorf("Add result was incorrect, got %v, expected %v", tensor1.GetData(), []float64{5.0, 16.0, 8.0})
	}
}
