from basalt import Tensor, TensorShape


@value
@register_passable("trivial")
struct Symbol(Copyable, Movable, Stringable, EqualityComparable):
    var name: UInt32
    var dtype: DType
    var shape: TensorShape
    var trainable: Bool

    fn __eq__(self, other: Self) -> Bool:
        return self.name == other.name

    fn __ne__(self, other: Self) -> Bool:
        return self.name != other.name

    fn __str__(self) -> String:
        return self.json()

    fn json(self) -> String:
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
