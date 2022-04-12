package tensor

import "fmt"

//Defines the representation of a tensor which will be used for all network operations
type Tensor struct {
	data []float64
}

//Creates a new tensor object and returns a reference to that object
func NewTensor(data []float64) *Tensor {
	return &Tensor{data: data}
}

//Getter for the data property of a tensor
func (tensor *Tensor) GetData() []float64 {
	return tensor.data
}

//In place element-wise addition of another tensor object to the current one - returns an error if subtraction could not be performed
func (tensor *Tensor) Add(other *Tensor) error {
	//Can only add tensors of the same size
	if len(tensor.data) != len(other.data) {
		return fmt.Errorf("cannot perform addition on two tensors of different dimensions, dimensions provided were: %d and %d", len(tensor.data), len(other.data))
	}

	for i := range tensor.data {
		tensor.data[i] += other.data[i]
	}

	return nil
}

//In place element-wise subtraction of another tensor object to the current one - returns an error if subtraction could not be performed
func (tensor *Tensor) Subtract(other *Tensor) error {
	//Can only subtract tensors of the same size
	if len(tensor.data) != len(other.data) {
		return fmt.Errorf("cannot perform subtraction on two tensors of different dimensions, dimensions provided were: %d and %d", len(tensor.data), len(other.data))
	}

	for i := range tensor.data {
		tensor.data[i] -= other.data[i]
	}

	return nil
}

//In Place multiplication of each element in the current tensor by each element in a different one - returns an error if the multiplication cannot be performed
func (tensor *Tensor) ElementMultiplication(other *Tensor) error {
	if len(tensor.data) != len(other.data) {
		return fmt.Errorf("cannot perform multiplication on two tensors of different dimensions, dimnensions provided were %d and %d", len(tensor.data), len(other.data))
	}

	for i := range tensor.data {
		tensor.data[i] *= other.data[i]
	}

	return nil
}

//Returns the dot product to two tensors - returns an error if dot product cannot be performed
func (tensor *Tensor) DotProduct(other *Tensor) (float64, error) {
	if len(tensor.data) != len(other.data) {
		return -1.0, fmt.Errorf("cannot perform dot product on two tensors of unequal dimensions, dimensions provided were: %d and %d", len(tensor.data), len(other.data))
	}

	var product float64 = 0
	for i, element := range tensor.data {
		product += element * other.data[i]
	}
	return product, nil
}
