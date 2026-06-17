from std.testing import assert_equal
from std.memory import memcpy

from basalt import dtype, nelts
from basalt import Tensor, TensorShape


def slice_rows[
    dtype: DType
](t: Tensor[dtype], start: Int, num_rows: Int) -> Tensor[dtype]:
    """
    Copies a contiguous range of leading-dimension rows out of `t`.

    Args:
        t: The tensor to slice, with the row dimension as its first axis.
        start: Index of the first row to copy.
        num_rows: Number of rows to copy.

    Returns:
        A new tensor of shape `(num_rows, *t.shape()[1:])` holding the
        copied rows.
    """
    var row_stride = t.strides()[0]
    var out_shape = t.shape()
    out_shape[0] = num_rows

    var out = Tensor[dtype](out_shape)
    memcpy(
        dest=out.mut_ptr(),
        src=t.ptr() + start * row_stride,
        count=num_rows * row_stride,
    )
    return out^


def cycle_pad_rows[
    dtype: DType
](t: Tensor[dtype], num_rows: Int) -> Tensor[dtype]:
    """
    Builds a tensor with `num_rows` leading-dimension rows by repeatedly
    cycling through `t`'s rows, used to pad a small held-out set up to a
    model's fixed batch size.

    Args:
        t: The tensor to cycle through, with the row dimension as its
            first axis.
        num_rows: Number of rows the returned tensor should have.

    Returns:
        A new tensor of shape `(num_rows, *t.shape()[1:])`.
    """
    var row_stride = t.strides()[0]
    var out_shape = t.shape()
    out_shape[0] = num_rows

    var out = Tensor[dtype](out_shape)
    for i in range(num_rows):
        var src_row = i % t.dim(0)
        memcpy(
            dest=out.mut_ptr() + i * row_stride,
            src=t.ptr() + src_row * row_stride,
            count=row_stride,
        )
    return out^


struct Batch[dtype: DType](Copyable, Movable):
    var data: Tensor[Self.dtype]
    var labels: Tensor[Self.dtype]

    def __init__(
        out self,
        batch_data: Tensor[Self.dtype],
        batch_labels: Tensor[Self.dtype],
    ):
        self.data = batch_data.copy()
        self.labels = batch_labels.copy()

    def __init__(
        out self,
        df_data: Tensor[Self.dtype],
        df_labels: Tensor[Self.dtype],
        start: Int,
        batch_data_shape: TensorShape,
        batch_labels_shape: TensorShape,
    ):
        # TODO: find a better way to do this
        # Links to the copies of the input tensors in model.forward()
        self.data = Tensor[Self.dtype](batch_data_shape)
        self.labels = Tensor[Self.dtype](batch_labels_shape)
        memcpy(
            dest=self.data.mut_ptr(),
            src=df_data.ptr() + (start * batch_data_shape.strides()[0]),
            count=batch_data_shape.num_elements(),
        )
        memcpy(
            dest=self.labels.mut_ptr(),
            src=df_labels.ptr() + (start * batch_labels_shape.strides()[0]),
            count=batch_labels_shape.num_elements(),
        )

    def __getitem__(self, index: Int) -> Tensor[Self.dtype]:
        if index == 0:
            return self.data.copy()
        elif index == 1:
            return self.labels.copy()
        else:
            print("[ERROR] Batch.__getitem__(): Index out of bounds")
            return Tensor[Self.dtype]()


struct DataLoader(Copyable, Movable):
    var data: Tensor[dtype]
    var labels: Tensor[dtype]
    var batch_size: Int
    var _current_index: Int
    var _num_batches: Int
    var _data_batch_shape: TensorShape
    var _label_batch_shape: TensorShape

    def __init__(
        out self,
        data: Tensor[dtype],
        labels: Tensor[dtype],
        batch_size: Int,
    ):
        self.data = data.copy()
        self.labels = labels.copy()
        self.batch_size = batch_size

        # Number of batches to iter, NOTE: ignore the remainder for now
        # var remainder = 1 if self.data.dim(0) % self.batch_size != 0 else 0
        self._current_index = 0
        self._num_batches = self.data.dim(0) // self.batch_size  # + remainder

        # Batch shapes
        self._data_batch_shape = self.data.shape()
        self._label_batch_shape = self.labels.shape()
        self._data_batch_shape[0] = self.batch_size
        self._label_batch_shape[0] = self.batch_size

    @always_inline
    def __len__(self) -> Int:
        """
        Returns the number of the batches left in the dataset.
        """
        return self._num_batches

    def __iter__(self) -> Self:
        # TODO: Starting the iterator requires to return (COPY!) the whole dataloader which containts the whole dataset
        # Does this mean that the whole dataset is copied every epoch ?!
        return self.copy()

    def __next__(mut self) raises StopIteration -> Batch[dtype]:
        if self._num_batches <= 0:
            raise StopIteration()
        var temp_current_index = self._current_index
        self._current_index += self.batch_size
        self._num_batches -= 1
        return Batch[dtype](
            self.data,
            self.labels,
            temp_current_index,
            self._data_batch_shape,
            self._label_batch_shape,
        )
