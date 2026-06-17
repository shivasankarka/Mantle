from std.testing import assert_equal, assert_almost_equal

from mantle import f32, nelts
from mantle.autograd import Graph, Symbol, OP
from mantle.nn import Model, Tensor, TensorShape, MSELoss, CrossEntropyLoss
from mantle.utils.tensorutils import fill


def test_MSE_perfect() raises:
    comptime y_pred_shape = TensorShape(2, 10)  # batch of 2, 10 classes
    comptime y_true_shape = TensorShape(2, 10)

    def create_graph() -> Graph:
        var g = Graph()

        var y_pred = g.input(y_pred_shape)
        var y_true = g.input(y_true_shape)

        var loss = MSELoss(g, y_pred, y_true)

        g.out(loss)

        return g ^

    comptime graph = create_graph()
    assert_equal(comptime(len(graph.nodes)), 3)

    var y_pred = Tensor[f32](y_pred_shape)
    var y_true = Tensor[f32](y_true_shape)

    fill(y_pred, 1.0)
    fill(y_true, 1.0)

    var model = Model[graph](inference_only=True)

    var outputs = model.inference(y_pred.share(), y_true.share())
    var loss = outputs[0].copy()

    assert_equal(loss.dim(0), 1)  # MSE summed over all elements
    assert_equal(loss[0], 0)  # loss is 0


def test_MSE_imperfect() raises:
    comptime y_pred_shape = TensorShape(1, 10)  # batch of 1, 10 classes
    comptime y_true_shape = TensorShape(1, 10)

    def create_graph() -> Graph:
        var g = Graph()

        var y_pred = g.input(y_pred_shape)
        var y_true = g.input(y_true_shape)

        var loss = MSELoss(g, y_pred, y_true)

        g.out(loss)

        return g ^

    comptime graph = create_graph()
    assert_equal(comptime(len(graph.nodes)), 3)

    var y_pred = Tensor[f32](y_pred_shape)
    var y_true = Tensor[f32](y_true_shape)

    fill(y_pred, 1.0)

    for i in range(10):
        y_true[i] = Float32(i)

    var model = Model[graph](inference_only=True)

    var outputs = model.inference(y_pred.share(), y_true.share())
    var loss = outputs[0].copy()

    var expected_loss: Scalar[f32] = 0.0

    for i in range(10):
        expected_loss += (y_pred[i] - y_true[i]) ** 2

    expected_loss = expected_loss / Float32(y_true_shape[1])

    assert_almost_equal(loss[0], expected_loss)


def test_CrossEntropy_perfect() raises:
    comptime y_pred_shape = TensorShape(2, 3)  # batch of 2, 3 classes
    comptime y_true_shape = TensorShape(2, 3)

    def create_graph() -> Graph:
        var g = Graph()

        var y_pred = g.input(y_pred_shape)
        var y_true = g.input(y_true_shape)

        var loss = CrossEntropyLoss(g, y_pred, y_true)

        g.out(loss)

        return g ^

    comptime graph = create_graph()
    assert_equal(comptime(len(graph.nodes)), 9)

    var y_pred = Tensor[f32](y_pred_shape)
    var y_true = Tensor[f32](y_true_shape)

    y_pred[0 * y_pred.dim(1) + 0] = 0.1
    y_pred[0 * y_pred.dim(1) + 1] = 0.2
    y_pred[0 * y_pred.dim(1) + 2] = 0.7
    y_true[0 * y_true.dim(1) + 0] = 0.0
    y_true[0 * y_true.dim(1) + 1] = 0.0
    y_true[0 * y_true.dim(1) + 2] = 1.0

    y_pred[1 * y_pred.dim(1) + 0] = 0.7
    y_pred[1 * y_pred.dim(1) + 1] = 0.2
    y_pred[1 * y_pred.dim(1) + 2] = 0.1
    y_true[1 * y_true.dim(1) + 0] = 1.0
    y_true[1 * y_true.dim(1) + 1] = 0.0
    y_true[1 * y_true.dim(1) + 2] = 0.0

    var model = Model[graph](inference_only=True)

    var outputs = model.inference(y_pred.copy(), y_true.copy())
    var loss = outputs[0].copy()

    assert_equal(loss.shape(), TensorShape(1))
    assert_almost_equal(loss[0], 0.76794958)


def test_CrossEntropy_imperfect() raises:
    comptime y_pred_shape = TensorShape(2, 3)  # batch of 2, 3 classes
    comptime y_true_shape = TensorShape(2, 3)

    def create_graph() -> Graph:
        var g = Graph()

        var y_pred = g.input(y_pred_shape)
        var y_true = g.input(y_true_shape)

        var loss = CrossEntropyLoss(g, y_pred, y_true)

        g.out(loss)

        return g ^

    comptime graph = create_graph()

    var y_pred = Tensor[f32](y_pred_shape)
    var y_true = Tensor[f32](y_true_shape)

    y_pred[0 * y_pred.dim(1) + 0] = 0.1
    y_pred[0 * y_pred.dim(1) + 1] = 0.2
    y_pred[0 * y_pred.dim(1) + 2] = 0.7
    y_true[0 * y_true.dim(1) + 0] = 0.0
    y_true[0 * y_true.dim(1) + 1] = 1.0
    y_true[0 * y_true.dim(1) + 2] = 0.0

    y_pred[1 * y_pred.dim(1) + 0] = 0.7
    y_pred[1 * y_pred.dim(1) + 1] = 0.2
    y_pred[1 * y_pred.dim(1) + 2] = 0.1
    y_true[1 * y_true.dim(1) + 0] = 0.0
    y_true[1 * y_true.dim(1) + 1] = 0.0
    y_true[1 * y_true.dim(1) + 2] = 1.0

    var model = Model[graph](inference_only=True)

    var outputs = model.inference(y_pred.copy(), y_true.copy())
    var loss = outputs[0].copy()

    assert_equal(loss.shape(), TensorShape(1))
    assert_almost_equal(loss[0], 1.31794953)


def main() raises:
    try:
        test_MSE_perfect()
        test_MSE_imperfect()
        test_CrossEntropy_perfect()
        test_CrossEntropy_imperfect()
    except e:
        print("[ERROR] Error in loss")
        print(e)
        raise e^
