from std.testing import assert_equal, assert_true
from std.utils.index import IndexList

from mantle.nn import TensorShape
from mantle.autograd.attributes import Attribute


def test_attribute_key() raises:
    comptime a = Attribute(name="test", value=-1)

    assert_true(String(a.name) == "test")


def test_attribute_int() raises:
    comptime value: Int = 1
    comptime a = Attribute(name="test", value=value)

    assert_true(a.to_int() == 1)


def test_attribute_string() raises:
    comptime value: String = "hello"
    comptime a = Attribute(name="test", value=value)

    assert_true(a.to_string() == value)


def test_attribute_tensor_shape() raises:
    comptime value: TensorShape = TensorShape(1, 2, 3)
    comptime a = Attribute(name="test", value=value)

    assert_true(a.to_shape() == value)


def test_attribute_static_int_tuple() raises:
    comptime value: IndexList[7] = IndexList[7](1, 2, 3, 4, 5, 6, 7)
    comptime a = Attribute(name="test", value=value)

    assert_true(a.to_static[7]() == value)


def test_attribute_scalar() raises:
    def test_float32() raises:
        comptime value_a: Float32 = 1.23456
        comptime a1 = Attribute(name="test", value=value_a)
        assert_true(
            a1.to_scalar[DType.float32]() == value_a,
            "Float32 scalar attribute failed",
        )

        comptime value_b: Float32 = 65151
        comptime a2 = Attribute(name="test", value=value_b)
        assert_true(
            a2.to_scalar[DType.float32]() == value_b,
            "Float32 scalar attribute failed",
        )

    def test_float_literal() raises:
        comptime value_c = -1.1
        comptime a3 = Attribute(name="test", value=value_c)
        assert_true(
            a3.to_scalar[DType.float32]() == value_c,
            "FloatLiteral scalar attribute failed",
        )

    def test_float64() raises:
        comptime value_a: Float64 = -1.23456
        comptime a1 = Attribute(name="test", value=value_a)
        assert_true(
            a1.to_scalar[DType.float64]() == value_a,
            "Float64 scalar attribute failed",
        )

        comptime value_b: Float64 = 123456
        comptime a2 = Attribute(name="test", value=value_b)
        assert_true(
            a2.to_scalar[DType.float64]() == value_b,
            "Float64 scalar attribute failed",
        )

    def test_int32() raises:
        comptime value_a: Int32 = 666
        comptime a1 = Attribute(name="test", value=value_a)
        assert_true(
            a1.to_scalar[DType.int32]() == value_a,
            "Int32 scalar attribute failed",
        )

        comptime value_b: Int32 = -666
        comptime a2 = Attribute(name="test", value=value_b)
        assert_true(
            a2.to_scalar[DType.int32]() == value_b,
            "Int32 scalar attribute failed",
        )

    def test_attribute_small_scalar() raises:
        comptime value_a: Float32 = 1e-18
        comptime a = Attribute(name="test", value=value_a)
        assert_true(
            a.to_scalar[DType.float32]() == value_a,
            "SMALL scalar attribute failed",
        )

    def test_attribute_big_scalar() raises:
        comptime value_a: Float32 = 1e40
        comptime a = Attribute(name="test", value=value_a)
        assert_true(
            a.to_scalar[DType.float32]() == value_a,
            "BIG scalar attribute failed",
        )

    test_float32()
    test_float_literal()
    test_float64()
    test_int32()
    test_attribute_small_scalar()
    test_attribute_big_scalar()


def main():
    try:
        test_attribute_key()
        test_attribute_int()
        test_attribute_string()
        test_attribute_tensor_shape()
        test_attribute_static_int_tuple()
        test_attribute_scalar()
    except e:
        print("[ERROR] Error in attributes")
        print(e)
