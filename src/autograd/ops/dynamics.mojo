from src import f32
from src.autograd.symbol import Symbol
from src.nn.tensor import Tensor, TensorShape
from src.utils.parameters import Parameters
from src.autograd.attributes import AttributeVector

from std.memory import memcpy


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
        attributes: AttributeVector
    ](inputs: List[Symbol], outputs: List[Symbol], mut parameters: Parameters,):
        comptime dim = attributes["dim"].value().to_int() if attributes[
            "dim"
        ] else 0
        var n_chunks = Self.calc_chunks(inputs[0].shape, dim)

        var chunks = List[Int]()
        var chunk_offsets: List[Int] = [0]
        for i in range(len(inputs)):
            chunks.append(inputs[i].shape.num_elements() // n_chunks)
            chunk_offsets.append(chunk_offsets[i] + chunks[i])

        var out_tensor = parameters.tensors[outputs[0]]
        for i in range(n_chunks):
            for j in range(len(inputs)):
                var in_tensor = parameters.tensors[inputs[j]]
                memcpy(
                    dest=out_tensor.mut_ptr()
                    + i * chunk_offsets[len(inputs)]
                    + chunk_offsets[j],
                    src=in_tensor.ptr() + i * chunks[j],
                    count=chunks[j],
                )

    @staticmethod
    def backward[
        input_id: Int, attributes: AttributeVector
    ](
        inputs: List[Symbol],
        outputs: List[Symbol],
        mut parameters: Parameters,
    ) -> Tensor[f32]:
        comptime dim = attributes["dim"].value().to_int() if attributes[
            "dim"
        ] else 0
        var n_chunks = Self.calc_chunks(inputs[0].shape, dim)

        var chunks = List[Int]()
        var chunk_offsets: List[Int] = [0]
        for i in range(len(inputs)):
            chunks.append(inputs[i].shape.num_elements() // n_chunks)
            chunk_offsets.append(chunk_offsets[i] + chunks[i])

        var res_grad = Tensor[f32](inputs[input_id].shape)
        var out_grad = parameters.grads[outputs[0]]
        for i in range(n_chunks):
            memcpy(
                dest=res_grad.mut_ptr() + i * chunks[input_id],
                src=out_grad.ptr()
                + i * chunk_offsets[len(inputs)]
                + chunk_offsets[input_id],
                count=chunks[input_id],
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
        attributes: AttributeVector
    ](inputs: List[Symbol], outputs: List[Symbol], mut parameters: Parameters,):
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

        var in_tensor = parameters.tensors[inputs[0]]
        for i in range(n_chunks):
            for j in range(len(outputs)):
                var out_tensor = parameters.tensors[outputs[j]]
                memcpy(
                    dest=out_tensor.mut_ptr() + i * chunks[j],
                    src=in_tensor.ptr()
                    + i * chunk_offsets[len(outputs)]
                    + chunk_offsets[j],
                    count=chunks[j],
                )

    @staticmethod
    def backward[
        input_id: Int, attributes: AttributeVector
    ](
        inputs: List[Symbol],
        outputs: List[Symbol],
        mut parameters: Parameters,
    ) -> Tensor[f32]:
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

        var res_grad = Tensor[f32](inputs[input_id].shape)

        for i in range(n_chunks):
            for j in range(len(outputs)):
                var out_grad = parameters.grads[outputs[j]]
                memcpy(
                    dest=res_grad.mut_ptr()
                    + i * chunk_offsets[len(outputs)]
                    + chunk_offsets[j],
                    src=out_grad.ptr() + i * chunks[j],
                    count=chunks[j],
                )

        return res_grad^
