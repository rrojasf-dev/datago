package main

import (
	"errors"
	"fmt"
)

// Tensor represents a basic tensor data structure
type Tensor struct {
	data  []int
	shape []int
}

// NewTensor creates a new tensor with the given shape and initializes it with the provided data.
func NewTensor(data []int, shape []int) (*Tensor, error) {
	expectedSize := 1
	for _, dim := range shape {
		expectedSize *= dim
	}
	if len(data) != expectedSize {
		return nil, errors.New("data size does not match shape")
	}
	return &Tensor{data: data, shape: shape}, nil
}

// Reshape reshapes the tensor to the new shape
func (t *Tensor) Reshape(newShape []int) (*Tensor, error) {
	expectedSize := 1
	for _, dim := range newShape {
		expectedSize *= dim
	}
	if expectedSize != len(t.data) {
		return nil, errors.New("new shape size mismatch with data size")
	}

	return &Tensor{data: t.data, shape: newShape}, nil
}

// HadamardProduct performs element-wise multiplication of two tensors
func (t *Tensor) HadamardProduct(other *Tensor) (*Tensor, error) {
	if len(t.shape) != len(other.shape) {
		return nil, errors.New("tensors must have the same number of dimensions")
	}
	for i := range t.shape {
		if t.shape[i] != other.shape[i] {
			return nil, errors.New("tensors must have the same shape")
		}
	}

	result := make([]int, len(t.data))
	for i := range t.data {
		result[i] = t.data[i] * other.data[i]
	}

	return &Tensor{data: result, shape: t.shape}, nil
}

// IndexSelect selects elements based on the specified dimension and indices
func (t *Tensor) IndexSelect(dim int, indices []int) (*Tensor, error) {
	if dim < 0 || dim >= len(t.shape) {
		return nil, errors.New("dimension out of range")
	}

	for _, index := range indices {
		if index < 0 || index >= t.shape[dim] {
			return nil, errors.New("index out of range")
		}
	}

	newShape := make([]int, len(t.shape))
	copy(newShape, t.shape)
	newShape[dim] = len(indices)

	stride := 1
	for i := dim + 1; i < len(t.shape); i++ {
		stride *= t.shape[i]
	}

	newData := make([]int, 0, len(indices)*stride)

	switch len(t.shape) {
	case 1:
		for _, index := range indices {
			newData = append(newData, t.data[index])
		}
	case 2:
		if dim == 0 {
			for _, index := range indices {
				start := index * t.shape[1]
				end := start + t.shape[1]
				newData = append(newData, t.data[start:end]...)
			}
		} else if dim == 1 {
			for i := 0; i < t.shape[0]; i++ {
				rowStart := i * t.shape[1]
				row := make([]int, len(indices))
				for j, index := range indices {
					row[j] = t.data[rowStart+index]
				}
				newData = append(newData, row...)
			}
		}
	default:
		return nil, errors.New("unsupported tensor rank")
	}

	return &Tensor{data: newData, shape: newShape}, nil
}

func main() {
	// Example tensor: [1, 2, 3, 4]
	data1 := []int{1, 2, 3, 4}
	shape1 := []int{4}
	tensor1, _ := NewTensor(data1, shape1)

	// Example tensor: [[1, 2], [3, 4]]
	data2 := []int{1, 2, 3, 4}
	shape2 := []int{2, 2}
	tensor2, _ := NewTensor(data2, shape2)

	// Example calls
	examples := []struct {
		Tensor   *Tensor
		Dimension int
		Indices  []int
	}{
		{tensor1, 0, []int{0, 0, 2}},
		{tensor2, 0, []int{0}},
		{tensor2, 0, []int{0, 0}},
		{tensor2, 0, []int{0, 0, 1, 1}},
		{tensor2, 1, []int{0}},
		{tensor2, 1, []int{0, 0}},
		{tensor2, 1, []int{0, 0, 1, 1}},
	}

	for _, example := range examples {
		result, err := example.Tensor.IndexSelect(example.Dimension, example.Indices)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
		} else {
			fmt.Printf("IndexSelect(%v, %d, %v) -> %v\n", example.Tensor, example.Dimension, example.Indices, result)
		}
	}

	// Example for Reshape
	reshaped, err := tensor2.Reshape([]int{4, 1})
	if err != nil {
		fmt.Println("Error:", err)
	} else {
		fmt.Printf("Reshape([[1, 2], [3, 4]], [4, 1]) -> %v\n", reshaped.data)
	}

	// Example for HadamardProduct
	data3 := []int{2, 0, 1, 2}
	shape3 := []int{2, 2}
	tensor3, _ := NewTensor(data3, shape3)
	product, _ := tensor2.HadamardProduct(tensor3)
	if err != nil {
		fmt.Println("Error:", err)
	} else {
		fmt.Printf("HadamardProduct([[1, 2], [3, 4]], [[2, 0], [1, 2]]) -> %v\n", product.data)
	}	
}


