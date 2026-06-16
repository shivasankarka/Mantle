from std.algorithm import vectorize, parallelize
from std.math import exp
from std.utils.numerics import min_finite, max_finite
from std.memory import memcpy

from basalt import Tensor, TensorShape
from basalt.utils.tensorutils import elwise_transform
from basalt.autograd.attributes import Attribute, AttributeVector


struct SIGMOID(Copyable, Movable):
    @staticmethod
    def result_shape(t1_shape: TensorShape) -> TensorShape:
        return t1_shape

    @staticmethod
    @always_inline
    def sigmoid[
        type: DType, simd_width: Int
    ](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
        return 1 / (1 + exp(-x))

    @staticmethod
    @always_inline
    def sidmoid_bw[
        type: DType, simd_width: Int
    ](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
        return Self.sigmoid(x) * (1 - Self.sigmoid(x))

    @staticmethod
    def forward[
        t1_shape: TensorShape,
    ](mut res: Tensor[dtype], t1: Tensor[dtype]):
        """Forward operation of sigmoid."""
        elwise_transform[Self.sigmoid](res, t1)

    @staticmethod
    def backward[
        ug_shape: TensorShape,
        t1_shape: TensorShape,
    ](ug: Tensor[dtype], t1: Tensor[dtype]) -> Tensor[dtype]:
        """Backward operation of sigmoid."""
        # d(sigmod(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        var res_grad = Tensor[dtype](ug_shape)

        def vec_sigmoid_bw[nelts: Int](idx: Int) {mut res_grad, read t1, read ug}:
            res_grad.store[nelts](
                idx,
                Self.sidmoid_bw(t1.load[nelts](idx)) * ug.load[nelts](idx),
            )

        vectorize[nelts](ug_shape.num_elements(), vec_sigmoid_bw)

        return res_grad^


struct RELU:
    @staticmethod
    def result_shape(t1_shape: TensorShape) -> TensorShape:
        return t1_shape

    @staticmethod
    @always_inline
    def relu[
        type: DType, simd_width: Int
    ](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
        # x if x > 0 else 0
        return (x > 0).select(x, 0)

    @staticmethod
    @always_inline
    def relu_bw[
        type: DType, simd_width: Int
    ](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
        # 1 if x > 0 else 0
        return (x > 0).select[type](1, 0)

    @staticmethod
    def forward[
        t1_shape: TensorShape,
    ](mut res: Tensor[dtype], t1: Tensor[dtype]):
        """Forward operation of relu."""
        elwise_transform[Self.relu](res, t1)

    @staticmethod
    def backward[
        ug_shape: TensorShape,
        t1_shape: TensorShape,
    ](ug: Tensor[dtype], t1: Tensor[dtype]) -> Tensor[dtype]:
        """Backward operation of relu."""
        # d(relu(x))/dx = 1 if x > 0 else 0. We also give 0 to x = 0 instead of undefined.
        var res_grad = Tensor[dtype](ug_shape)

        def vec_relu_bw[nelts: Int](idx: Int) {mut res_grad, read t1, read ug}:
            res_grad.store[nelts](
                idx, Self.relu_bw(t1.load[nelts](idx)) * ug.load[nelts](idx)
            )

        vectorize[nelts](ug_shape.num_elements(), vec_relu_bw)

        return res_grad^


struct LEAKYRELU:
    @staticmethod
    def result_shape(t1_shape: TensorShape) -> TensorShape:
        return t1_shape

    @staticmethod
    def forward[
        t1_shape: TensorShape,
        attributes: AttributeVector,
    ](mut res: Tensor[dtype], t1: Tensor[dtype]):
        """Forward operation of leaky_relu."""

        def leaky_relu[
            type: DType,
            simd_width: Int,
        ](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
            var negative_slope = (
                attributes["negative_slope"].value().to_scalar[type]()
            )
            return (x > 0).select(x, x * negative_slope)

        elwise_transform[leaky_relu](res, t1)

    @staticmethod
    def backward[
        ug_shape: TensorShape,
        t1_shape: TensorShape,
        attributes: AttributeVector,
    ](ug: Tensor[dtype], t1: Tensor[dtype]) -> Tensor[dtype]:
        """Backward operation of leaky_relu."""

        @always_inline
        def leaky_relu_bw[
            type: DType, simd_width: Int
        ](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
            var negative_slope = (
                attributes["negative_slope"].value().to_scalar[type]()
            )

            return (x > 0).select[type](1, negative_slope)

        var res_grad = Tensor[dtype](ug_shape)

        def vec_leaky_relu_bw[nelts: Int](idx: Int) {mut res_grad, read t1, read ug}:
            res_grad.store[nelts](
                idx,
                leaky_relu_bw(t1.load[nelts](idx)) * ug.load[nelts](idx),
            )

        vectorize[nelts](ug_shape.num_elements(), vec_leaky_relu_bw)

        return res_grad^


struct TANH:
    @staticmethod
    def result_shape(t1_shape: TensorShape) -> TensorShape:
        return t1_shape

    @staticmethod
    @always_inline
    def tanh[
        type: DType, simd_width: Int
    ](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    @staticmethod
    @always_inline
    def tanh_bw[
        type: DType, simd_width: Int
    ](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
        return 1 - pow(Self.tanh(x), 2)

    @staticmethod
    def forward[
        t1_shape: TensorShape,
    ](mut res: Tensor[dtype], t1: Tensor[dtype]):
        """Forward operation of tanh."""
        elwise_transform[Self.tanh](res, t1)

    @staticmethod
    def backward[
        ug_shape: TensorShape,
        t1_shape: TensorShape,
    ](ug: Tensor[dtype], t1: Tensor[dtype]) -> Tensor[dtype]:
        """Backward operation of tanh."""
        # d(tanh(x))/dx = 1 - tanh(x) ** 2
        var res_grad = Tensor[dtype](ug_shape)

        def vec_tanh_bw[nelts: Int](idx: Int) {mut res_grad, read t1, read ug}:
            res_grad.store[nelts](
                idx, Self.tanh_bw(t1.load[nelts](idx)) * ug.load[nelts](idx)
            )

        vectorize[nelts](ug_shape.num_elements(), vec_tanh_bw)

        return res_grad^


struct CLIP:
    @staticmethod
    def result_shape(t_shape: TensorShape) -> TensorShape:
        return t_shape

    @staticmethod
    def forward[
        t_shape: TensorShape, attributes: AttributeVector
    ](mut res: Tensor[dtype], t: Tensor[dtype]):
        """
        Forward pass of the clip operation.
        """
        comptime min_attr = attributes["min"]
        comptime max_attr = attributes["max"]

        var min_val = min_attr.value().to_scalar[
            dtype
        ]() if min_attr else min_finite[dtype]()
        var max_val = max_attr.value().to_scalar[
            dtype
        ]() if max_attr else max_finite[dtype]()

        def vec_clip[nelts: Int](i: Int) {mut res, read t, read min_val, read max_val}:
            res.store[nelts](i, max(min(t.load[nelts](i), max_val), min_val))

        vectorize[nelts](t_shape.num_elements(), vec_clip)

    @staticmethod
    def backward[
        ug_shape: TensorShape,
        t_shape: TensorShape,
        attributes: AttributeVector = AttributeVector(),
    ](ug: Tensor[dtype], t: Tensor[dtype]) -> Tensor[dtype]:
        """Backward operation of clip."""
        comptime min_attr = attributes["min"]
        comptime max_attr = attributes["max"]

        var min_val = min_attr.value().to_scalar[
            dtype
        ]() if min_attr else min_finite[dtype]()
        var max_val = max_attr.value().to_scalar[
            dtype
        ]() if max_attr else max_finite[dtype]()

        var res_grad = Tensor[dtype](t_shape)

        def vec_clip_bw[nelts: Int](i: Int) {mut res_grad, read t, read ug, read min_val, read max_val}:
            var val = t.load[nelts](i)
            res_grad.store[nelts](
                i,
                ((val >= min_val) * (val <= max_val)).select(
                    ug.load[nelts](i), 0
                ),
            )

        vectorize[nelts](t_shape.num_elements(), vec_clip_bw)

        return res_grad^


struct SQUEEZE:
    @staticmethod
    def result_shape(
        t1_shape: TensorShape, attributes: AttributeVector
    ) -> TensorShape:
        var dim = attributes["dims"]
        var dims_to_squeeze = dim.value().to_shape() if dim else TensorShape()

        var new_shape = List[Int]()
        for i in range(t1_shape.rank()):
            if (not dim and t1_shape[i] == 1) or (
                i in dims_to_squeeze and t1_shape[i] == 1
            ):
                continue
            new_shape.append(t1_shape[i])

        return TensorShape(new_shape)

    @staticmethod
    def forward[
        t1_shape: TensorShape,
        attributes: AttributeVector,
    ](mut res: Tensor[dtype], t1: Tensor[dtype]):
        memcpy(res.data(), t1.data(), t1.num_elements())

    @staticmethod
    def backward[
        ug_shape: TensorShape,
        t1_shape: TensorShape,
    ](ug: Tensor[dtype], t1: Tensor[dtype]) -> Tensor[dtype]:
        var res_grad = Tensor[dtype](t1_shape)
        memcpy(res_grad.data(), ug.data(), ug.num_elements())
        return res_grad^


struct UNSQUEEZE:
    @staticmethod
    def result_shape(
        t1_shape: TensorShape, attributes: AttributeVector
    ) -> TensorShape:
        var dim = attributes["dims"]
        var dims_to_squeeze = dim.value().to_shape() if dim else TensorShape()

        # Position in the expanded dims where the new dim (or dims) is placed.
        var new_rank = t1_shape.rank() + dims_to_squeeze.rank()

        var new_shape = List[Int]()
        var j = 0
        for i in range(new_rank):
            if i in dims_to_squeeze or i - new_rank in dims_to_squeeze:
                new_shape.append(1)
            else:
                new_shape.append(t1_shape[j])
                j += 1

        return TensorShape(new_shape)

    @staticmethod
    def forward[
        t1_shape: TensorShape,
        attributes: AttributeVector,
    ](mut res: Tensor[dtype], t1: Tensor[dtype]):
        memcpy(res.data(), t1.data(), t1.num_elements())

    @staticmethod
    def backward[
        ug_shape: TensorShape,
        t1_shape: TensorShape,
    ](ug: Tensor[dtype], t1: Tensor[dtype]) -> Tensor[dtype]:
        var res_grad = Tensor[dtype](t1_shape)
        memcpy(res_grad.data(), ug.data(), ug.num_elements())
        return res_grad^


struct SLICE:
    @staticmethod
    def adjust_boundary(slice: Int, dim_size: Int) -> Int:
        # Adjust negative indices & ensure they are within bounds.
        var s = slice if slice >= 0 else dim_size + slice
        return max(min(s, dim_size), 0)

    @staticmethod
    def default_starts(shape: List[Int]) -> List[Int]:
        var starts = List[Int]()
        for i in range(len(shape)):
            starts.append(0)
        return starts^

    @staticmethod
    def default_ends(shape: List[Int]) -> List[Int]:
        var ends = List[Int]()
        for i in range(len(shape)):
            ends.append(shape[i])
        return ends^

    @staticmethod
    def default_steps(shape: List[Int]) -> List[Int]:
        var steps = List[Int]()
        for i in range(len(shape)):
            steps.append(1)
        return steps^

    @staticmethod
    def default_axes(shape: TensorShape) -> List[Int]:
        # NOTE: axes can't be negative
        var axes = List[Int]()
        for i in range(shape.rank()):
            axes.append(i)
        return axes^

    @staticmethod
    def result_shape(
        t1_shape: TensorShape, attributes: AttributeVector
    ) -> TensorShape:
        # NOTE: Starts and ends have to be of the same size
        # NOTE: If axes not provided, starts and ends have to be of the same size as t1_shape
        var starts = attributes["starts"].value().to_list()
        var ends = attributes["ends"].value().to_list()
        var steps = (
            attributes["steps"]
            .value()
            .to_list() if attributes["steps"] else Self.default_steps(starts)
        )
        var axes = (
            attributes["axes"]
            .value()
            .to_list() if attributes["axes"] else Self.default_axes(t1_shape)
        )

        var new_shape = t1_shape
        for i in range(len(starts)):
            var axis = axes[i]
            new_shape[axis] = len(
                range(
                    start=Self.adjust_boundary(starts[i], t1_shape[axis]),
                    end=Self.adjust_boundary(ends[i], t1_shape[axis]),
                    step=steps[i],
                )
            )

        return new_shape

    @staticmethod
    def reorder_positions[
        id: Int
    ](original: List[Int], axes: List[Int], t1_shape: List[Int]) -> List[Int]:
        # Reorder the starts (id=0), ends (id=1) or steps (id=2) to match the order of the axes
        var updated: List[Int]

        comptime if id == 0:
            updated = Self.default_starts(t1_shape)
        elif id == 1:
            updated = Self.default_ends(t1_shape)
        else:
            updated = Self.default_steps(t1_shape)

        for i in range(len(axes)):
            var axis = axes[i]
            updated[axis] = original[i] if id == 2 else Self.adjust_boundary(
                original[i], t1_shape[axis]
            )

        return updated^

    # NOTE: For now you can't have recursive function as parameter functions.
    # NOTE: From testing it seems a recursive function is almost the same speed as doing multiple nested for loops.
    @staticmethod
    def recursive_iters_slice[
        shape: TensorShape,
        original_shape: TensorShape,
        steps: List[Int],
        starts: List[Int],
        ends: List[Int],
        backward_op: Bool = False,
    ](
        mut res: Tensor[dtype],
        t1: Tensor[dtype],
        last_dims: Int,
        position: Int,
        last_position: Int,
        idx: Int,
        idx_original: Int,
    ):
        comptime strides = shape.strides()
        comptime t1_strides = original_shape.strides()

        var idx_temp = idx
        var idx_original_temp = (
            starts[position] * t1_strides[position] + idx_original
        )

        if position == last_position + 1:
            # Work on the last dimensions
            comptime position = shape.rank() - 1
            comptime stride = t1_strides[position] * steps[position]

            def v_slice[nelts: Int](k: Int) {mut res, read t1, mut idx_original_temp, read idx_temp}:
                comptime if not backward_op:
                    comptime if steps[position] == 1:
                        res.store[nelts](
                            idx_temp + k, t1.load[nelts](idx_original_temp)
                        )
                    else:
                        res.store[nelts](
                            idx_temp + k,
                            t1.data()
                            .offset(idx_original_temp)
                            .strided_load[width=nelts](stride),
                        )
                else:
                    comptime if steps[position] == 1:
                        res.store[nelts](
                            idx_original_temp, t1.load[nelts](idx_temp + k)
                        )
                    else:
                        res.data().offset(idx_original_temp).strided_store[
                            width=nelts
                        ](t1.load[nelts](idx_temp + k), stride)

                idx_original_temp += stride * nelts

            vectorize[nelts](last_dims, v_slice)

            return

        for _ in range(shape[position]):
            Self.recursive_iters_slice[
                shape, original_shape, steps, starts, ends, backward_op
            ](
                res,
                t1,
                last_dims,
                position + 1,
                last_position,
                idx_temp,
                idx_original_temp,
            )

            idx_temp += strides[position]
            idx_original_temp += steps[position] * t1_strides[position]

    @staticmethod
    def slice_kernel[
        res_shape: TensorShape,
        original_shape: TensorShape,
        steps: List[Int],
        starts: List[Int],
        ends: List[Int],
        backward_op: Bool = False,
    ](mut res: Tensor[dtype], t1: Tensor[dtype]):
        comptime strides = original_shape.strides()

        # Get the dimensions for vectorization
        var last_dims = 1
        var positions_to_skip = 0
        for i in range(res_shape.rank() - 1, -1, -1):
            if steps[i] != 1 and i != res_shape.rank() - 1:
                break
            last_dims *= res_shape[i]
            positions_to_skip += 1
            if starts[i] != 0 or ends[i] != original_shape[i] or steps[i] != 1:
                break

        # Get the dimensions for the first loop
        var first_dims = 1
        var start_position = 0
        for i in range(res_shape.rank() - positions_to_skip):
            if steps[i] != 1 or starts[i] != 0 or ends[i] != original_shape[i]:
                break
            first_dims *= res_shape[i]
            start_position += 1

        var middle_dims = res_shape.num_elements() // last_dims // first_dims

        @parameter
        def p_slice(i: Int):
            Self.recursive_iters_slice[
                res_shape, original_shape, steps, starts, ends, backward_op
            ](
                res,
                t1,
                last_dims,
                start_position,
                res_shape.rank() - 1 - positions_to_skip,
                i * middle_dims * last_dims,
                i * strides[start_position - 1],
            )

        parallelize[p_slice](first_dims)

    @staticmethod
    def forward[
        t1_shape: TensorShape,
        attributes: AttributeVector,
    ](mut res: Tensor[dtype], t1: Tensor[dtype]):
        comptime axes = attributes["axes"].value().to_list() if attributes[
            "axes"
        ] else Self.default_axes(t1_shape)
        comptime starts = Self.reorder_positions[0](
            attributes["starts"].value().to_list(), axes, t1_shape.to_list()
        )
        comptime ends = Self.reorder_positions[1](
            attributes["ends"].value().to_list(), axes, t1_shape.to_list()
        )
        comptime steps = Self.reorder_positions[2](
            attributes["steps"].value().to_list(), axes, t1_shape.to_list()
        ) if attributes["steps"] else Self.default_steps(t1_shape.to_list())

        comptime res_shape = Self.result_shape(t1_shape, attributes)

        Self.slice_kernel[res_shape, t1_shape, steps, starts, ends, False](
            res, t1
        )

    @staticmethod
    def backward[
        ug_shape: TensorShape,
        t1_shape: TensorShape,
        attributes: AttributeVector = AttributeVector(),
    ](ug: Tensor[dtype], t1: Tensor[dtype]) -> Tensor[dtype]:
        comptime axes = attributes["axes"].value().to_list() if attributes[
            "axes"
        ] else Self.default_axes(t1_shape)
        comptime starts = Self.reorder_positions[0](
            attributes["starts"].value().to_list(), axes, t1_shape.to_list()
        )
        comptime ends = Self.reorder_positions[1](
            attributes["ends"].value().to_list(), axes, t1_shape.to_list()
        )
        comptime steps = Self.reorder_positions[2](
            attributes["steps"].value().to_list(), axes, t1_shape.to_list()
        ) if attributes["steps"] else Self.default_steps(t1_shape.to_list())

        var res_grad = Tensor[dtype](t1_shape)

        Self.slice_kernel[ug_shape, t1_shape, steps, starts, ends, True](
            res_grad, ug
        )

        return res_grad^
