from basalt import Tensor, TensorShape

# I don't know what's the shape doing yet and why there's no initializer


struct Symbol(
    Equatable, ImplicitlyCopyable, Movable, TrivialRegisterPassable, Writable
):
    var name: UInt32  # name of the symbol
    var dtype: DType
    var shape: TensorShape
    var trainable: Bool

    def __eq__(self, other: Self) -> Bool:
        return self.name == other.name

    def __ne__(self, other: Self) -> Bool:
        return self.name != other.name

    def __str__(self) -> String:
        return self.json()

    def json(self) -> String:
        return (
            '{"name": "'
            + String(self.name)
            + '", "dtype": "'
            + String(self.dtype)
            + '", "shape": "'
            + String(self.shape)
            + '", "trainable": "'
            + String(self.trainable)
            + '"}'
        )
