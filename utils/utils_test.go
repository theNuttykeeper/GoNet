package utils

import "testing"

//Tests the FloatSlicesEqual method that compares whether two slices are equal
func TestFloatSlicesEqual(t *testing.T) {
	slice1 := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	slice2 := []float64{3.0, 4.0, 5.0, 6.0, 7.0}
	slice3 := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	slice4 := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}

	if !FloatSlicesEqual(slice1, slice3) {
		t.Errorf("floatsliceequals method incorrectly detected slices as different, slices were: %v, %v", slice1, slice3)
	}

	if FloatSlicesEqual(slice1, slice2) {
		t.Errorf("floatsliceequals method incorrectly detected slices as equal, slices were: %v, %v", slice1, slice2)
	}

	if FloatSlicesEqual(slice1, slice4) {
		t.Errorf("floatsliceequals method incorrectly detected slices of different lengths as equal, slices were: %v, %v", slice1, slice4)
	}
}
