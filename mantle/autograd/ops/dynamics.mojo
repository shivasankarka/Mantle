# ===----------------------------------------------------------------------=== #
# Mantle: Dynamic Ops
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
#  ===----------------------------------------------------------------------=== #
"""Dynamic Ops (mantle.autograd.ops.dynamics)
------------------------------------------------
Variable-input/output operators (CONCAT, SPLIT) with forward and backward passes.
"""
from mantle import f32
from mantle.autograd.symbol import Symbol
from mantle.core.tensor import Tensor, TensorShape
from mantle.core.device import Device
from mantle.nn.parameters import Parameters
from mantle.autograd.attributes import AttributeVector
from .gpu_elementwise import gpu_strided_block_copy

from std.memory import unsafe_memcpy


# ===----------------------------------------------------------------------===#
# CONCAT
# ===----------------------------------------------------------------------===#


struct CONCAT:
    @staticmethod
    def result_shape(
        input_shapes: List[TensorShape], attributes: AttributeVector
    ) -> List[TensorShape]:
        # Assumptions: all tensors have the same shape, except for the concatenating dimension
        var dim = attributes["dim"].value().to_int() if attributes["dim"] else 0

        var concat_size: Int = 0
        for i in range(len(input_shapes)):
            concat_size += input_shapes[i][dim]

        var res_shape = input_shapes[0]
        res_shape[dim] = concat_size

        return [res_shape]

    @staticmethod
    def calc_chunks(shape: TensorShape, dim: Int) -> Int:
        # Number of chunks up to the concatenating dimension
        # Assuming tensor of equal shape, except for the concatenating dimension
        var chunks = 1
        for i in range(dim):
            chunks *= shape[i]
        return chunks

    @staticmethod
    def forward[
        attributes: AttributeVector, device: Device = Device.cpu
    ](
        inputs: List[Symbol],
        outputs: List[Symbol],
        mut parameters: Parameters[device],
    ) raises:
        comptime dim = attributes["dim"].value().to_int() if attributes[
            "dim"
        ] else 0
        var n_chunks = Self.calc_chunks(inputs[0].shape, dim)

        var chunks = List[Int]()
        var chunk_offsets: List[Int] = [0]
        for i in range(len(inputs)):
            chunks.append(inputs[i].shape.num_elements() // n_chunks)
            chunk_offsets.append(chunk_offsets[i] + chunks[i])

        comptime if device.id == Device.cpu.id:
            var out_tensor = rebind[Parameters[Device.cpu]](
                parameters
            ).tensors[outputs[0]]
            for i in range(n_chunks):
                for j in range(len(inputs)):
                    var in_tensor = rebind[Parameters[Device.cpu]](
                        parameters
                    ).tensors[inputs[j]]
                    unsafe_memcpy(
                        dest=out_tensor.ptr().unsafe_offset(
                            i * chunk_offsets[len(inputs)] + chunk_offsets[j]
                        ),
                        src=in_tensor.ptr().unsafe_offset(i * chunks[j]),
                        count=chunks[j],
                    )
        else:
            var out_tensor = rebind[Parameters[Device.gpu]](
                parameters
            ).tensors[outputs[0]]
            for j in range(len(inputs)):
                var in_tensor = rebind[Parameters[Device.gpu]](
                    parameters
                ).tensors[inputs[j]]
                gpu_strided_block_copy(
                    out_tensor,
                    in_tensor,
                    n_chunks=n_chunks,
                    count=chunks[j],
                    src_chunk_stride=chunks[j],
                    dst_chunk_stride=chunk_offsets[len(inputs)],
                    src_offset=0,
                    dst_offset=chunk_offsets[j],
                )

    @staticmethod
    def backward[
        input_id: Int, attributes: AttributeVector, device: Device = Device.cpu
    ](
        inputs: List[Symbol],
        outputs: List[Symbol],
        mut parameters: Parameters[device],
    ) raises -> Tensor[f32, device]:
        comptime dim = attributes["dim"].value().to_int() if attributes[
            "dim"
        ] else 0
        var n_chunks = Self.calc_chunks(inputs[0].shape, dim)

        var chunks = List[Int]()
        var chunk_offsets: List[Int] = [0]
        for i in range(len(inputs)):
            chunks.append(inputs[i].shape.num_elements() // n_chunks)
            chunk_offsets.append(chunk_offsets[i] + chunks[i])

        var res_grad = Tensor[f32, device](inputs[input_id].shape)
        comptime if device.id == Device.cpu.id:
            var out_grad = rebind[Parameters[Device.cpu]](parameters).grads[
                outputs[0]
            ]
            for i in range(n_chunks):
                unsafe_memcpy(
                    dest=rebind[Tensor[f32, Device.cpu]](
                        res_grad
                    ).ptr().unsafe_offset(i * chunks[input_id]),
                    src=out_grad.ptr().unsafe_offset(
                        i * chunk_offsets[len(inputs)]
                        + chunk_offsets[input_id]
                    ),
                    count=chunks[input_id],
                )
        else:
            var out_grad = rebind[Parameters[Device.gpu]](parameters).grads[
                outputs[0]
            ]
            gpu_strided_block_copy(
                rebind[Tensor[f32, Device.gpu]](res_grad),
                out_grad,
                n_chunks=n_chunks,
                count=chunks[input_id],
                src_chunk_stride=chunk_offsets[len(inputs)],
                dst_chunk_stride=chunks[input_id],
                src_offset=chunk_offsets[input_id],
                dst_offset=0,
            )

        return res_grad^


struct SPLIT:
    @staticmethod
    def result_shape(
        input_shapes: List[TensorShape], attributes: AttributeVector
    ) -> List[TensorShape]:
        # Assuming the sum of the sections is equal to the total size in the dim dimension.
        # E.g. sections = [5, 5, 2] -> shape (., 12, ., .) for dim = 1
        var dim = attributes["dim"].value().to_int() if attributes["dim"] else 0
        var sections = attributes["sections"].value().to_shape()

        var res_shapes = List[TensorShape]()
        for i in range(sections.rank()):
            var new_shape = input_shapes[0]
            new_shape[dim] = sections[i]
            res_shapes.append(new_shape)

        return res_shapes^

    @staticmethod
    def calc_chunks(shape: TensorShape, dim: Int) -> Int:
        # Number of chunks up to the concatenating dimension
        # Assuming tensor of equal shape, except for the concatenating dimension
        var chunks = 1
        for i in range(dim):
            chunks *= shape[i]
        return chunks

    @staticmethod
    def forward[
        attributes: AttributeVector, device: Device = Device.cpu
    ](
        inputs: List[Symbol],
        outputs: List[Symbol],
        mut parameters: Parameters[device],
    ) raises:
        comptime dim = attributes["dim"].value().to_int() if attributes[
            "dim"
        ] else 0
        comptime sections = attributes["sections"].value().to_shape()
        var n_chunks = Self.calc_chunks(inputs[0].shape, dim)

        var chunks = List[Int]()
        var chunk_offsets: List[Int] = [0]
        for i in range(len(outputs)):
            chunks.append(outputs[i].shape.num_elements() // n_chunks)
            chunk_offsets.append(chunk_offsets[i] + chunks[i])

        comptime if device.id == Device.cpu.id:
            var in_tensor = rebind[Parameters[Device.cpu]](
                parameters
            ).tensors[inputs[0]]
            for i in range(n_chunks):
                for j in range(len(outputs)):
                    var out_tensor = rebind[Parameters[Device.cpu]](
                        parameters
                    ).tensors[outputs[j]]
                    unsafe_memcpy(
                        dest=out_tensor.ptr().unsafe_offset(i * chunks[j]),
                        src=in_tensor.ptr().unsafe_offset(
                            i * chunk_offsets[len(outputs)] + chunk_offsets[j]
                        ),
                        count=chunks[j],
                    )
        else:
            var in_tensor = rebind[Parameters[Device.gpu]](
                parameters
            ).tensors[inputs[0]]
            for j in range(len(outputs)):
                var out_tensor = rebind[Parameters[Device.gpu]](
                    parameters
                ).tensors[outputs[j]]
                gpu_strided_block_copy(
                    out_tensor,
                    in_tensor,
                    n_chunks=n_chunks,
                    count=chunks[j],
                    src_chunk_stride=chunk_offsets[len(outputs)],
                    dst_chunk_stride=chunks[j],
                    src_offset=chunk_offsets[j],
                    dst_offset=0,
                )

    @staticmethod
    def backward[
        input_id: Int, attributes: AttributeVector, device: Device = Device.cpu
    ](
        inputs: List[Symbol],
        outputs: List[Symbol],
        mut parameters: Parameters[device],
    ) raises -> Tensor[f32, device]:
        comptime dim = attributes["dim"].value().to_int() if attributes[
            "dim"
        ] else 0
        comptime sections = attributes["sections"].value().to_shape()
        var n_chunks = Self.calc_chunks(inputs[0].shape, dim)

        var chunks: List[Int] = []
        var chunk_offsets: List[Int] = [0]
        for i in range(len(outputs)):
            chunks.append(outputs[i].shape.num_elements() // n_chunks)
            chunk_offsets.append(chunk_offsets[i] + chunks[i])

        var res_grad = Tensor[f32, device](inputs[input_id].shape)

        comptime if device.id == Device.cpu.id:
            for i in range(n_chunks):
                for j in range(len(outputs)):
                    var out_grad = rebind[Parameters[Device.cpu]](
                        parameters
                    ).grads[outputs[j]]
                    unsafe_memcpy(
                        dest=rebind[Tensor[f32, Device.cpu]](
                            res_grad
                        ).ptr().unsafe_offset(
                            i * chunk_offsets[len(outputs)] + chunk_offsets[j]
                        ),
                        src=out_grad.ptr().unsafe_offset(i * chunks[j]),
                        count=chunks[j],
                    )
        else:
            for j in range(len(outputs)):
                var out_grad = rebind[Parameters[Device.gpu]](
                    parameters
                ).grads[outputs[j]]
                gpu_strided_block_copy(
                    rebind[Tensor[f32, Device.gpu]](res_grad),
                    out_grad,
                    n_chunks=n_chunks,
                    count=chunks[j],
                    src_chunk_stride=chunks[j],
                    dst_chunk_stride=chunk_offsets[len(outputs)],
                    src_offset=0,
                    dst_offset=chunk_offsets[j],
                )

        return res_grad^
