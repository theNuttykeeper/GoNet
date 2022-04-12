package main

import (
	"fmt"

	"github.com/theNuttykeeper/GoNet/layers"
	//"github.com/theNuttykeeper/GoNet/tensor"
)

func main() {
	fmt.Println("This is the GoNet Neural Network Package")

	layer := layers.NewLinear(8, 4)
	fmt.Println(layer.Details())
}
