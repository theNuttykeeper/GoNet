package tensor

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

//In place element-wise addition of another tensor object to the current one
func (tensor *Tensor) Add(other *Tensor) {
	//Can only add tensors of the same size
	if len(tensor.data) != len(other.data) {
		return
	}

	for i, element := range tensor.data {
		tensor.data[i] = element + other.data[i]
	}
}
