# ===----------------------------------------------------------------------=== #
# Mantle: DataLoader
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""DataLoader (mantle.data.dataloader)
------------------------------------------------
Mini-batch iteration and row-slicing utilities.
"""
from std.testing import assert_equal
from std.memory import unsafe_memcpy
from std.random import random_ui64

from mantle import f32, nelts
from mantle.core.tensor import Tensor, TensorShape


# ===----------------------------------------------------------------------===#
# Slice Rows
# ===----------------------------------------------------------------------===#


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
    unsafe_memcpy(
        dest=out.ptr(),
        src=t.ptr().unsafe_offset(start * row_stride),
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
        unsafe_memcpy(
            dest=out.ptr().unsafe_offset(i * row_stride),
            src=t.ptr().unsafe_offset(src_row * row_stride),
            count=row_stride,
        )
    return out^


# ===----------------------------------------------------------------------===#
# Batch
# ===----------------------------------------------------------------------===#


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
        unsafe_memcpy(
            dest=self.data.ptr(),
            src=df_data.ptr().unsafe_offset(
                start * batch_data_shape.strides()[0]
            ),
            count=batch_data_shape.num_elements(),
        )
        unsafe_memcpy(
            dest=self.labels.ptr(),
            src=df_labels.ptr().unsafe_offset(
                start * batch_labels_shape.strides()[0]
            ),
            count=batch_labels_shape.num_elements(),
        )

    def __init__(
        out self,
        df_data: Tensor[Self.dtype],
        df_labels: Tensor[Self.dtype],
        indices: List[Int],
        start: Int,
        batch_data_shape: TensorShape,
        batch_labels_shape: TensorShape,
    ):
        """Copy a batch selected by a shuffled row permutation."""
        self.data = Tensor[Self.dtype](batch_data_shape)
        self.labels = Tensor[Self.dtype](batch_labels_shape)
        var data_row_stride = batch_data_shape.strides()[0]
        var label_row_stride = batch_labels_shape.strides()[0]
        var source_data_row_stride = df_data.strides()[0]
        var source_label_row_stride = df_labels.strides()[0]
        for row in range(batch_data_shape[0]):
            var source_row = indices[start + row]
            unsafe_memcpy(
                dest=self.data.ptr().unsafe_offset(row * data_row_stride),
                src=df_data.ptr().unsafe_offset(
                    source_row * source_data_row_stride
                ),
                count=data_row_stride,
            )
            unsafe_memcpy(
                dest=self.labels.ptr().unsafe_offset(row * label_row_stride),
                src=df_labels.ptr().unsafe_offset(
                    source_row * source_label_row_stride
                ),
                count=label_row_stride,
            )

    def __getitem__(self, index: Int) -> Tensor[Self.dtype]:
        if index == 0:
            return self.data.copy()
        elif index == 1:
            return self.labels.copy()
        else:
            print("[ERROR] Batch.__getitem__(): Index out of bounds")
            return Tensor[Self.dtype]()


# ===----------------------------------------------------------------------===#
# DataLoader
# ===----------------------------------------------------------------------===#


struct DataLoaderIterator(Movable, Copyable):
    """An epoch iterator that shares dataset storage with its loader."""

    var data: Tensor[f32]
    var labels: Tensor[f32]
    var batch_size: Int
    var _current_index: Int
    var _num_batches: Int
    var _data_batch_shape: TensorShape
    var _label_batch_shape: TensorShape
    var _indices: List[Int]

    def __init__(
        out self,
        var data: Tensor[f32],
        var labels: Tensor[f32],
        batch_size: Int,
        drop_last: Bool,
        shuffle: Bool,
        seed: UInt64,
    ):
        self.data = data^
        self.labels = labels^
        self.batch_size = batch_size
        self._current_index = 0
        self._num_batches = self.data.dim(0) // self.batch_size
        if not drop_last and self.data.dim(0) % self.batch_size != 0:
            self._num_batches += 1

        self._data_batch_shape = self.data.shape()
        self._label_batch_shape = self.labels.shape()
        self._data_batch_shape[0] = self.batch_size
        self._label_batch_shape[0] = self.batch_size
        self._indices = List[Int]()
        if shuffle:
            self._indices.reserve(self.data.dim(0))
            for i in range(self.data.dim(0)):
                self._indices.append(i)
            var state = seed
            if state == 0:
                state = 88172645463325252
            for i in range(self.data.dim(0) - 1, 0, -1):
                state = state ^ (state << 13)
                state = state ^ (state >> 7)
                state = state ^ (state << 17)
                var j = Int(state % UInt64(i + 1))
                var value = self._indices[i]
                self._indices[i] = self._indices[j]
                self._indices[j] = value

    def __iter__(self) -> Self:
        return self.copy()

    def __next__(mut self) raises StopIteration -> Batch[f32]:
        if self._num_batches <= 0:
            raise StopIteration()
        var start = self._current_index
        var rows = min(self.batch_size, self.data.dim(0) - start)
        self._current_index += rows
        self._num_batches -= 1

        var data_shape = self._data_batch_shape.copy()
        var label_shape = self._label_batch_shape.copy()
        data_shape[0] = rows
        label_shape[0] = rows
        if len(self._indices) == 0:
            return Batch[f32](self.data, self.labels, start, data_shape, label_shape)
        return Batch[f32](
            self.data,
            self.labels,
            self._indices,
            start,
            data_shape,
            label_shape,
        )


struct DataLoader(Copyable, Movable):
    var data: Tensor[f32]
    var labels: Tensor[f32]
    var batch_size: Int
    var drop_last: Bool
    var shuffle: Bool
    var seed: UInt64
    var _current_index: Int
    var _num_batches: Int
    var _data_batch_shape: TensorShape
    var _label_batch_shape: TensorShape

    def __init__(
        out self,
        data: Tensor[f32],
        labels: Tensor[f32],
        batch_size: Int,
        drop_last: Bool = True,
        shuffle: Bool = False,
        seed: Int = -1,
    ):
        self.data = data.copy()
        self.labels = labels.copy()
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = random_ui64(0, UInt64.MAX) if seed < 0 else UInt64(seed)

        self._current_index = 0
        self._num_batches = self.data.dim(0) // self.batch_size
        if not self.drop_last and self.data.dim(0) % self.batch_size != 0:
            self._num_batches += 1

        # Batch shapes
        self._data_batch_shape = self.data.shape()
        self._label_batch_shape = self.labels.shape()
        self._data_batch_shape[0] = self.batch_size
        self._label_batch_shape[0] = self.batch_size

    @always_inline
    def __len__(self) -> Int:
        """
        Returns the number of batches in one epoch.
        """
        return self._num_batches

    def __iter__(self) -> DataLoaderIterator:
        """Start a fresh epoch without copying the dataset buffers."""
        return DataLoaderIterator(
            self.data.share(),
            self.labels.share(),
            self.batch_size,
            self.drop_last,
            self.shuffle,
            self.seed,
        )

    def reset(mut self):
        """Restart iteration without reconstructing the loader."""
        self._current_index = 0
        self._num_batches = self.data.dim(0) // self.batch_size
        if not self.drop_last and self.data.dim(0) % self.batch_size != 0:
            self._num_batches += 1

    def __next__(mut self) raises StopIteration -> Batch[f32]:
        if self._num_batches <= 0:
            raise StopIteration()
        var temp_current_index = self._current_index
        var rows = self.batch_size
        var remaining = self.data.dim(0) - temp_current_index
        if remaining < rows:
            rows = remaining
        self._current_index += rows
        self._num_batches -= 1

        var data_shape = self._data_batch_shape.copy()
        var label_shape = self._label_batch_shape.copy()
        data_shape[0] = rows
        label_shape[0] = rows
        return Batch[f32](
            self.data,
            self.labels,
            temp_current_index,
            data_shape,
            label_shape,
        )
