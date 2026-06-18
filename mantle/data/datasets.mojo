from std.algorithm import vectorize

from mantle import f32
from mantle.core.tensor import Tensor, TensorShape
from mantle.core.tensorutils import elwise_op, tmean, tstd


@always_inline
def div[
    dtype: DType, simd_width: Int
](a: SIMD[dtype, simd_width], b: Scalar[dtype]) -> SIMD[dtype, simd_width]:
    return a / b


struct BostonHousing:
    comptime n_inputs = 13

    var data: Tensor[f32]
    var labels: Tensor[f32]

    def __init__(out self, file_path: String) raises:
        var s = open(file_path, "r").read()
        # Skip the first and last lines
        # This does assume your last line in the file has a newline at the end
        var all_lines = s.split("\n")
        var list_of_lines = all_lines[1 : len(all_lines) - 1]

        # Length is number of lines
        var N = len(list_of_lines)

        self.data = Tensor[f32](
            N, self.n_inputs
        )  # All columns except the last one
        self.labels = Tensor[f32](N, 1)  # Only the last column (MEDV)

        # Load data in Tensor
        for item in range(N):
            var line = list_of_lines[item].split(",")
            # var line_strings = List[String](line)
            self.labels[item] = cast_string[f32](String(line[len(line) - 1]))

            for n in range(self.n_inputs):
                self.data[item * self.n_inputs + n] = cast_string[f32](
                    String(line[n])
                )

        # Normalize data
        # TODO: redo when tensorutils tmean2 and tstd2 are implemented
        comptime nelts = simd_width_of[f32]()
        var col = Tensor[f32](N)
        for j in range(self.n_inputs):
            for k in range(N):
                col[k] = self.data[k * self.n_inputs + j]
            for i in range(N):
                self.data[i * self.n_inputs + j] = (
                    self.data[i * self.n_inputs + j] - tmean(col)
                ) / tstd(col)


struct MNIST:
    var data: Tensor[f32]
    var labels: Tensor[f32]

    def __init__(out self, file_path: String) raises:
        # var s = read_file(file_path)
        var s = open(file_path, "r").read()
        # Skip the first and last lines
        # This does assume your last line in the file has a newline at the end
        var all_lines = s.split("\n")
        var list_of_lines = all_lines[1 : len(all_lines) - 1]

        # Length is number of lines
        var N = len(list_of_lines)
        self.data = Tensor[f32](N, 1, 28, 28)
        self.labels = Tensor[f32](N)

        # Load data in Tensor
        for item in range(N):
            var line = list_of_lines[item].split(",")
            self.labels[item] = Scalar[f32](atol(String(line[0])))
            for i in range(self.data.shape()[2]):
                for j in range(self.data.shape()[3]):
                    self.data[item * 28 * 28 + i * 28 + j] = Scalar[f32](
                        atol(String(line[i * 28 + j + 1]))
                    )

        # Normalize data
        comptime nelts = simd_width_of[f32]()

        def vecdiv[nelts: Int](idx: Int) {mut self}:
            self.data.store[nelts](idx, div(self.data.load[nelts](idx), 255.0))

        vectorize[nelts](self.data.num_elements(), vecdiv)


def read_file(file_path: String) raises -> String:
    var s: String
    with open(file_path, "r") as f:
        s = f.read()
    return s


def find_first(s: String, delimiter: String) -> Int:
    for i in range(s.byte_length()):
        if String(s[byte=i]) == delimiter:
            return i
    return -1


def cast_string[dtype: DType](s: String) raises -> Scalar[dtype]:
    """
    Cast a string with decimal to a SIMD vector of dtype.
    """

    var idx = find_first(s, delimiter=".")

    if idx == -1:
        # No decimal point
        return Scalar[dtype](atol(s))
    else:
        var c_int = Scalar[dtype](atol(s[byte=:idx]))
        var c_frac = Scalar[dtype](atol(s[byte = idx + 1 :]))
        return c_int + c_frac / Scalar[dtype](
            10 ** s[byte = idx + 1 :].byte_length()
        )
