"""
`Parameters`: the runtime tensor/gradient storage used by `Model` and read
by every `forward_op`/`backward_op`.
"""

from mantle.utils.collection import Collection


struct Parameters:
    var tensors: Collection
    var grads: Collection

    def __init__(out self):
        self.tensors = Collection()
        self.grads = Collection()
