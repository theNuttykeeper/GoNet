package tensor

import "fmt"

//Enum class containing all the possible types a tensor can take
type TensorType int64

const (
	Integer TensorType = iota
	Float
)

//Converts the tensorType into a string representation
func (tensorType TensorType) String() string {
	switch tensorType {
	case Integer:
		return "Integer"
	case Float:
		return "Float"
	}
	return "Unknown"
}

//Representation of a Tensor
type Tensor struct {
	tensorType TensorType
}

//Displays all the details about the tensor
func (tensor *Tensor) Details() string {
	return fmt.Sprintf("Type: %v", tensor.tensorType.String())
}

//Creates a new object of the Tensor struct and returns a pointer to that object
func NewTensor(tensorType TensorType) *Tensor {
	return &Tensor{tensorType: tensorType}
}
