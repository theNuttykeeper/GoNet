package main

import (
	"fmt"

	//"github.com/theNuttykeeper/GoNet/layers"
	"github.com/theNuttykeeper/GoNet/tensor"
)

func main() {
	fmt.Println("This is the GoNet Neural Network Package")
	tensor := tensor.NewTensor(tensor.Float)
	fmt.Println(tensor.Details())
}
