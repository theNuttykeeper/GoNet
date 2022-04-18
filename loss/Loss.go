package loss

import (
	"fmt"
	"math"

	"github.com/theNuttykeeper/GoNet/tensor"
)

//Function which calculates the Mean Squared Error loss for a given output
func MSELoss(output, expected *tensor.Tensor) (float64, error) {
	if len(output.GetData()) != len(expected.GetData()) {
		return 0.0, fmt.Errorf("expected and actual values not the same size, got sizes %d and %d", len(output.GetData()), len(expected.GetData()))
	}

	loss := 0.0
	expectedData := expected.GetData()
	for i, element := range output.GetData() {
		loss += math.Pow((element - expectedData[i]), 2)
	}

	loss /= float64(len(expectedData))

	return loss, nil
}
