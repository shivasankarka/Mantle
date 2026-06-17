from std.math import sqrt
from std.algorithm import vectorize, parallelize

from .model import Parameters
from basalt import Graph, Tensor, TensorShape
from basalt.utils.collection import Collection
from basalt.utils.math_util import add, sub, mul, div


def get_trainable_parameters(g: Graph) -> List[Symbol]:
    """
    Get all symbols of trainable parameters.
    """

    var trainable_parameters = List[Symbol]()

    for i in range(len(g.params)):
        if g.params.symbols[i].trainable:
            trainable_parameters.append(g.params.symbols[i])

    return trainable_parameters^


struct Adam[
    g: Graph,
    trainable_parameters: List[Symbol] = get_trainable_parameters(g),
]:
    var parameters: Pointer[Parameters, MutAnyOrigin]

    var lr: Scalar[f32]
    var beta1: Scalar[f32]
    var beta2: Scalar[f32]
    var epsilon: Scalar[f32]
    var iter: Int

    var rms_grads: Collection
    var momentum_grads: Collection

    def __init__(
        out self,
        ref[MutAnyOrigin] parameters: Parameters,
        lr: Scalar[f32] = 0.001,
        beta1: Scalar[f32] = 0.9,
        beta2: Scalar[f32] = 0.999,
        epsilon: Scalar[f32] = 1e-8,
    ):
        # self.parameters = Pointer.address_of(parameters)
        self.parameters = Pointer(to=parameters)

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iter = 0

        var tr = materialize[Self.trainable_parameters]()
        # Capacity of the collections should be the n of trainable parameters
        self.rms_grads = Collection(capacity=len(tr))
        self.momentum_grads = Collection(capacity=len(tr))

        self.allocate_rms_and_momentum()

    def zero_grad(mut self):
        """Set all gradients to zero."""
        self.parameters[].grads.set_zero()

    def step(mut self):
        """Update model parameters."""
        self.iter += 1
        var tr = materialize[Self.trainable_parameters]()

        # Loop over all trainable parameters
        @parameter
        def p_step(i: Int):
            var param = tr[i]

            def v_step[nelts: Int](j: Int) {mut self, read param}:
                var momentum_grads = self.momentum_grads[param].load[nelts](j)
                var rms_grads = self.rms_grads[param].load[nelts](j)
                var grads = self.parameters[].grads[param].load[nelts](j)
                var params = self.parameters[].tensors[param].load[nelts](j)

                # Momentum beta 1
                # f1 = beta1 * momentum + (1 - beta1) * grad
                momentum_grads = (
                    self.beta1 * momentum_grads + (1 - self.beta1) * grads
                )
                self.momentum_grads[param].store[nelts](j, momentum_grads)

                # Bias correction
                # f2 = f1 / (1 - beta1 ** iter)
                momentum_grads = momentum_grads / (1 - self.beta1**self.iter)

                # RMS beta 2
                # f1 = beta2 * rms + (1 - beta2) * grad ** 2
                rms_grads = (
                    self.beta2 * rms_grads + (1 - self.beta2) * grads * grads
                )
                self.rms_grads[param].store[nelts](j, rms_grads)

                # Bias correction
                # f2 = f1 / (1 - beta2 ** iter)
                rms_grads = rms_grads / (1 - self.beta2**self.iter)

                # tensor = tensor - lr * (f2 / (sqrt(rms) + epsilon))
                params = params - self.lr * (
                    momentum_grads / (sqrt(rms_grads) + self.epsilon)
                )
                self.parameters[].tensors[param].store[nelts](j, params)

            vectorize[1](param.shape.num_elements(), v_step)

        parallelize[p_step](len(tr))

    def allocate_rms_and_momentum(mut self):
        # They are initialized to zero
        # Loop over all trainable parameters
        var tr = materialize[Self.trainable_parameters]()
        for i in range(len(tr)):
            var param = tr[i]
            self.rms_grads.append(Tensor[f32](param.shape), param)
            self.momentum_grads.append(Tensor[f32](param.shape), param)
