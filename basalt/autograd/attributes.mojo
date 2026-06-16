from std.collections import Optional, OptionalReg
from std.utils.static_tuple import StaticTuple
from std.utils.index import IndexList

from basalt.nn.tensor import Tensor, TensorShape, MAX_RANK
from basalt.utils.bytes import Bytes, scalar_to_bytes, bytes_to_scalar

# Figure out what this attribute does in this code.
# 1) AttributeType seems to be defining some container for the type being used with fixed size
#
#
#
#


comptime MAX_ATTRS = 10
comptime MAX_NAME_CHARS = 16
comptime MAX_DATA_BYTES = 32


struct AttributeType(TrivialRegisterPassable, Writable):
    comptime BOOL = AttributeType(0, "BOOL")
    comptime INT = AttributeType(1, "INT")
    comptime FLOAT = AttributeType(2, "FLOAT")
    comptime STRING = AttributeType(3, "STRING")
    comptime INTS = AttributeType(4, "INTS")
    comptime FLOATS = AttributeType(5, "FLOATS")

    var id: UInt8
    var name: Bytes[MAX_NAME_CHARS]  #

    def __init__(out self, id: UInt8, name: String):
        self.id = id
        self.name = Bytes[MAX_NAME_CHARS](name)

    def __init__(out self, type: DType):
        if type.is_floating_point():
            self = AttributeType.FLOAT
        elif type == DType.bool:
            self = AttributeType.BOOL
        else:
            self = AttributeType.INT

    def __eq__(self, other: Self) -> Bool:
        return self.id == other.id

    def __str__(self) -> String:
        return String(self.name)


struct AttributeVector(
    Copyable, Movable, Sized, TrivialRegisterPassable, Writable
):
    var attributes: StaticTuple[Attribute, MAX_ATTRS]
    var size: Int

    def __init__(out self, *attributes: Attribute):
        self.attributes = StaticTuple[Attribute, MAX_ATTRS](Attribute("", ""))
        self.size = len(attributes)
        for i in range(self.size):
            self.attributes[i] = attributes[i]

    @always_inline("nodebug")
    def __len__(self) -> Int:
        return self.size

    @always_inline("nodebug")
    def __getitem__(self, index: Int) -> Attribute:
        return self.attributes[index]

    @always_inline("nodebug")
    def __getitem__(self, index: StringLiteral) -> OptionalReg[Attribute]:
        for i in range(self.size):
            if self.attributes[i].name == Bytes[MAX_NAME_CHARS](index):
                return self.attributes[i]
        return None

    def __str__(self) -> String:
        var s: String = "["
        for i in range(self.size):
            s += String(self.attributes[i])
            if i < self.size - 1:
                s += ", "
        return s + "]"


struct Attribute(Copyable, Movable, TrivialRegisterPassable, Writable):
    var data_shape: IndexList[MAX_RANK]
    var name: Bytes[MAX_NAME_CHARS]
    var data: Bytes[MAX_DATA_BYTES]
    var type: AttributeType
    var size: Int

    def __init__(out self, name: String, value: String):
        self.data_shape = IndexList[MAX_RANK]()
        self.name = Bytes[MAX_NAME_CHARS](name)
        self.data = Bytes[MAX_DATA_BYTES](value)
        self.type = AttributeType.STRING
        self.size = (
            value.byte_length()
        )  # check if it should be byte_length or codepoints.

    def __init__(out self, name: String, value: TensorShape):
        self.data_shape = IndexList[MAX_RANK]()
        self.name = Bytes[MAX_NAME_CHARS](name)
        self.data = Bytes[MAX_DATA_BYTES]()
        self.type = AttributeType.INTS
        self.size = value.rank()

        for i in range(self.size):
            self.data_shape[i] = value._shape[i]

    def __init__[N: Int](out self, name: String, value: IndexList[N]):
        comptime assert (
            N < MAX_RANK
        ), "Attribute rank must be less than MAX_RANK."
        self.data_shape = IndexList[MAX_RANK]()
        self.name = Bytes[MAX_NAME_CHARS](name)
        self.data = Bytes[MAX_DATA_BYTES]()
        self.type = AttributeType.INTS
        self.size = N

        for i in range(self.size):
            self.data_shape[i] = value[i]

    def __init__(out self, name: String, value: List[Int]):
        self.data_shape = IndexList[MAX_RANK]()
        self.name = Bytes[MAX_NAME_CHARS](name)
        self.data = Bytes[MAX_DATA_BYTES]()
        self.type = AttributeType.INTS
        self.size = len(value)

        for i in range(self.size):
            self.data_shape[i] = value[i]

    def __init__(out self, name: String, value: StaticTuple[Int, _]):
        self.data_shape = IndexList[MAX_RANK]()
        self.name = Bytes[MAX_NAME_CHARS](name)
        self.data = Bytes[MAX_DATA_BYTES]()
        self.type = AttributeType.INTS
        self.size = len(value)

        for i in range(self.size):
            self.data_shape[i] = value[i]

    def __init__[dtype: DType](out self, name: String, value: Scalar[dtype]):
        comptime assert dtype.is_numeric(), "Attribute value must be numeric."

        self.data_shape = IndexList[MAX_RANK]()
        self.name = Bytes[MAX_NAME_CHARS](name)
        self.data = scalar_to_bytes[dtype, MAX_DATA_BYTES](value)
        self.type = AttributeType(dtype)
        self.size = 1

    def __init__(out self, name: String, value: Int):
        self = Self.__init__(name, Int64(value))
        self.data_shape[0] = 1

    def __init__(out self, name: String, value: FloatLiteral):
        self = Self.__init__(name, Float64(value))
        self.data_shape[0] = 1

    @always_inline("nodebug")
    def __str__(self) -> String:
        return "Attribute(" + String(self.name) + ", " + "..." + ")"

    @always_inline("nodebug")
    def to_string(self) -> String:
        return String(self.data)

    @always_inline("nodebug")
    def to_list(self) -> List[Int]:
        var result = List[Int]()

        for i in range(self.size):
            result.append(self.data_shape[i])

        return result^

    @always_inline("nodebug")
    def to_shape(self) -> TensorShape:
        return TensorShape(rank=self.size, shape=self.data_shape)

    @always_inline("nodebug")
    def to_static[N: Int](self) -> IndexList[N]:
        comptime assert (
            N < MAX_RANK
        ), "Attribute rank must be less than MAX_RANK."

        var result = IndexList[N]()
        for i in range(N):
            result[i] = Int(self.data_shape[i])

        return result

    @always_inline("nodebug")
    def to_scalar[dtype: DType](self) -> Scalar[dtype]:
        comptime assert dtype.is_numeric(), "Attribute value must be numeric."

        return bytes_to_scalar[dtype](self.data)

    @always_inline("nodebug")
    def to_int(self) -> Int:
        return Int(self.to_scalar[DType.int64]())

    def json(self) -> String:
        var result = '{"name": "' + String(self.name) + '", '

        var type: String = ""
        var value: String = ""

        if self.type == AttributeType.STRING:
            type = "STRING"
            value = '"' + self.to_string() + '"'
        elif self.type == AttributeType.INTS:
            type = "INTS"

            var value_temp = self.to_shape()
            value = "["
            for i in range(value_temp.rank()):
                value += String(value_temp._shape[i])
                if i < value_temp.rank() - 1:
                    value += ", "
            value += "]"
        elif self.type == AttributeType.FLOAT:
            type = "FLOAT"
            value = String(self.to_scalar[DType.float64]())
        elif self.type == AttributeType.INT:
            type = "INT"
            value = String(self.to_int())
        else:
            type = "UNKNOWN"
            value = "UNKNOWN"

        result += '"type": "' + type + '", ' + '"value": ' + value

        return result + "}"
