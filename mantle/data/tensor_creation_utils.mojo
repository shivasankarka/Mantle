from std.python import Python, PythonObject
from std.memory import memcpy, UnsafePointer

from mantle import f32
from mantle.core.tensor import Tensor, TensorShape

# maybe this functions should be from the Tensor struct (like tensor.to_numpy()) and tensor.__init__(np_array: PythonObject) to create a tensor from a numpy array and tensor.copy_np_data(np_array: PythonObject) to copy the numpy array to the tensor.


def to_numpy[dtype: DType](tensor: Tensor[dtype]) -> PythonObject:
    try:
        var np = Python.import_module("numpy")

        np.set_printoptions(4)

        var rank = tensor.rank()
        var dims = Python.list()
        for i in range(rank):
            dims.append(tensor.dim(i))
        var pyarray: PythonObject = np.empty(dims, dtype=np.float32)

        var pointer_d = pyarray.__array_interface__["data"][
            0
        ].unsafe_get_as_pointer[DType.float32]()
        var d = tensor.ptr().bitcast[Float32]()
        memcpy(dest=pointer_d, src=d, count=tensor.num_elements())

        _ = tensor

        return pyarray^
    except e:
        print("Error in to numpy", e)
        return PythonObject()


def to_tensor(np_array: PythonObject) raises -> Tensor[f32]:
    var shape = List[Int]()
    for i in range(Int(py=np_array.ndim)):
        shape.append(Int(py=np_array.shape[i]))
    if np_array.ndim == 0:
        # When the numpy array is a scalar, you need or the reshape to a size 1 ndarray or do this, if not the memcpy gets a memory error (Maybe because it is a register value?).
        var tensor = Tensor[f32](TensorShape(1))
        tensor[0] = Scalar[f32](py=np_array)
        return tensor^

    var tensor = Tensor[f32](TensorShape(shape))

    var np_array_2: PythonObject
    try:
        var np = Python.import_module("numpy")
        # copy is also necessary for ops like slices to make them contiguous instead of references.
        np_array_2 = np.float32(np_array.copy())
    except e:
        np_array_2 = np_array.copy()
        print("Error in to_tensor", e)

    var pointer_d = np_array_2.__array_interface__["data"][
        0
    ].unsafe_get_as_pointer[f32]()
    memcpy(dest=tensor.mut_ptr(), src=pointer_d, count=tensor.num_elements())

    _ = np_array_2
    _ = np_array

    return tensor^


def copy_np_data[dtype: DType](mut tensor: Tensor[dtype], np_array: PythonObject) raises:
    var np_array_2: PythonObject
    try:
        var np = Python.import_module("numpy")
        # copy is also necessary for ops like slices to make them contiguous instead of references.
        np_array_2 = np.float32(np_array.copy())
    except e:
        np_array_2 = np_array.copy()
        print("Error in to_tensor", e)

    var pointer_d = np_array_2.__array_interface__["data"][
        0
    ].unsafe_get_as_pointer[dtype]()
    memcpy(dest=tensor.mut_ptr(), src=pointer_d, count=tensor.num_elements())

    # This shouldn't be necessary anymore, but I'm leaving it here for now.
    # _ = np_array_2
    # _ = np_array
    # _ = tensor
