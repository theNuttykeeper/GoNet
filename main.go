package main

import (
	"fmt"

	//"github.com/theNuttykeeper/GoNet/layers"
	"github.com/theNuttykeeper/GoNet/tensor"
)

func main() {
	fmt.Println("This is the GoNet Neural Network Package")

	tensor1 := tensor.NewTensor([]float64{1.2, 10.0, 5.0})
	tensor2 := tensor.NewTensor([]float64{4.0, 6.0, 3.0})
	tensor1.Add(tensor2)
	fmt.Println(tensor1)
	fmt.Println(tensor2)
}
