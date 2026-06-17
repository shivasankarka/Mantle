"""
Native binary checkpoint format for saving and restoring `Model`/`Adam`
state, keyed by graph `Symbol` id.
"""

from std.memory import UnsafePointer
from std.sys.info import size_of

from basalt import Symbol, Tensor, TensorShape, f32
from basalt.nn.model import Parameters
from basalt.utils.collection import Collection

# ===----------------------------------------------------------------------=== #
# File format constants
# ===----------------------------------------------------------------------=== #

comptime CHECKPOINT_MAGIC = UInt32(0x42534C54)
"""Magic number ("BSLT" packed as a little-endian u32) at the start of every checkpoint file."""

comptime CHECKPOINT_VERSION = UInt32(1)
"""Checkpoint file format version. Bumped on incompatible layout changes."""

comptime OPTIM_KIND_NONE = UInt8(0)
"""Entry kind tag for a plain model tensor (not optimizer state)."""

comptime OPTIM_KIND_MOMENTUM = UInt8(1)
"""Entry kind tag for an `Adam.momentum_grads` entry."""

comptime OPTIM_KIND_RMS = UInt8(2)
"""Entry kind tag for an `Adam.rms_grads` entry."""

# ===----------------------------------------------------------------------=== #
# Byte-level helpers
# ===----------------------------------------------------------------------=== #


def _write_u32(mut f: FileHandle, value: UInt32) raises:
    """
    Writes a `UInt32` to `f` as 4 raw little-endian bytes.

    Args:
        f: The file handle to write to.
        value: The value to write.

    Raises:
        Any error raised by the underlying file write.
    """
    var v = value
    var bytes = Span[UInt8](ptr=UnsafePointer(to=v).bitcast[UInt8](), length=4)
    f.write_bytes(bytes)


def _read_u32(data: List[UInt8], mut offset: Int) -> UInt32:
    """
    Reads a `UInt32` from `data` at `offset` as 4 raw little-endian bytes.

    Args:
        data: The byte buffer to read from.
        offset: Byte position to start reading at. Advanced by 4 in place.

    Returns:
        The decoded value.
    """
    var v: UInt32 = 0
    var bytes = Span[UInt8](ptr=UnsafePointer(to=v).bitcast[UInt8](), length=4)
    for i in range(4):
        bytes[i] = data[offset + i]
    offset += 4
    return v


def _write_tensor(
    mut f: FileHandle, symbol_id: UInt32, kind: UInt8, tensor: Tensor[f32]
) raises:
    """
    Writes one checkpoint entry: symbol id, kind tag, shape, then the
    tensor's raw element bytes.

    Args:
        f: The file handle to write to.
        symbol_id: The graph `Symbol.name` this tensor is keyed by.
        kind: One of `OPTIM_KIND_NONE`/`OPTIM_KIND_MOMENTUM`/`OPTIM_KIND_RMS`.
        tensor: The tensor whose data should be written.

    Raises:
        Any error raised by the underlying file write.
    """
    _write_u32(f, symbol_id)
    var kind_byte = kind
    f.write_bytes(
        Span[UInt8](ptr=UnsafePointer(to=kind_byte).bitcast[UInt8](), length=1)
    )
    _write_u32(f, UInt32(tensor.rank()))
    for i in range(tensor.rank()):
        _write_u32(f, UInt32(tensor.dim(i)))

    var n_bytes = tensor.num_elements() * size_of[f32]()
    var data_bytes = Span[UInt8](ptr=tensor.ptr().bitcast[UInt8](), length=n_bytes)
    f.write_bytes(data_bytes)


def _read_entry_header(
    data: List[UInt8], mut offset: Int
) -> Tuple[UInt32, UInt8, TensorShape]:
    """
    Reads one checkpoint entry's header (symbol id, kind tag, shape),
    leaving `offset` positioned at the start of the entry's raw data bytes.

    Args:
        data: The byte buffer to read from.
        offset: Byte position to start reading at. Advanced in place.

    Returns:
        A `(symbol_id, kind, shape)` tuple describing the entry.
    """
    var symbol_id = _read_u32(data, offset)
    var kind = data[offset]
    offset += 1
    var rank = Int(_read_u32(data, offset))

    var dims = List[Int]()
    for _ in range(rank):
        dims.append(Int(_read_u32(data, offset)))

    return (symbol_id, kind, TensorShape(dims))


def _read_tensor_into(data: List[UInt8], mut offset: Int, mut collection: Collection):
    """
    Reads one tensor entry and copies its bytes into `collection`, keyed
    by the saved symbol id.

    Args:
        data: The byte buffer to read from.
        offset: Byte position to start reading at. Advanced in place.
        collection: The tensor collection to copy the entry's data into.

    Notes:
        If the id isn't present in `collection` (e.g. the checkpoint was
        made from a different graph), the bytes are skipped so the read
        cursor stays in sync with the rest of the file.
    """
    var header = _read_entry_header(data, offset)
    var symbol_id = header[0]
    var shape = header[2]
    var n_bytes = shape.num_elements() * size_of[f32]()

    var index = collection.get_index(symbol_id)
    if index == -1:
        offset += n_bytes
        return

    ref tensor = collection.data_ref[index]
    var out_ptr = tensor.mut_ptr().bitcast[UInt8]()
    for i in range(n_bytes):
        out_ptr[i] = data[offset + i]
    offset += n_bytes


# ===----------------------------------------------------------------------=== #
# Public API
# ===----------------------------------------------------------------------=== #


struct CheckpointInfo(Copyable, Movable):
    """
    Metadata recovered from a checkpoint file, returned by
    `load_checkpoint`/`load_checkpoint_with_optim` so the caller knows
    what was restored.
    """

    var iter: Int
    """The optimizer step count stored in the checkpoint (0 if not set)."""

    var has_optim_state: Bool
    """Whether the checkpoint file contained Adam momentum/rms entries."""

    def __init__(out self, iter: Int, has_optim_state: Bool):
        """
        Args:
            iter: The optimizer step count stored in the checkpoint.
            has_optim_state: Whether the file contains optimizer state.
        """
        self.iter = iter
        self.has_optim_state = has_optim_state


def save_checkpoint(path: String, parameters: Parameters, iter: Int = 0) raises:
    """
    Serializes model tensors to a flat native binary file, keyed by
    symbol id.

    Notes:
        Use `save_checkpoint_with_optim` instead if you also want to
        preserve Adam's momentum/rms state across the save.

    Args:
        path: Destination file path.
        parameters: The model's tensors to serialize.
        iter: Optimizer step count to record in the file (purely
            informational when no optimizer state is saved).

    Raises:
        Any error raised by the underlying file write.
    """
    var f = open(path, "w")

    _write_u32(f, CHECKPOINT_MAGIC)
    _write_u32(f, CHECKPOINT_VERSION)
    _write_u32(f, UInt32(iter))
    _write_u32(f, UInt32(len(parameters.tensors)))
    _write_u32(f, 0)  # num_optim entries

    for i in range(len(parameters.tensors)):
        _write_tensor(
            f,
            parameters.tensors.symbols_ref[i],
            OPTIM_KIND_NONE,
            parameters.tensors.data_ref[i],
        )

    f.close()


def save_checkpoint_with_optim(
    path: String,
    parameters: Parameters,
    momentum_grads: Collection,
    rms_grads: Collection,
    iter: Int = 0,
) raises:
    """
    Like `save_checkpoint`, but also serializes Adam's per-parameter
    momentum/rms state so a resumed training run continues with correct
    optimizer statistics instead of a cold start.

    Args:
        path: Destination file path.
        parameters: The model's tensors to serialize.
        momentum_grads: `Adam.momentum_grads` to serialize alongside the
            model tensors.
        rms_grads: `Adam.rms_grads` to serialize alongside the model
            tensors.
        iter: Optimizer step count to record in the file (`Adam.iter`).

    Raises:
        Any error raised by the underlying file write.
    """
    var f = open(path, "w")

    _write_u32(f, CHECKPOINT_MAGIC)
    _write_u32(f, CHECKPOINT_VERSION)
    _write_u32(f, UInt32(iter))

    var num_momentum = len(momentum_grads)
    var num_rms = len(rms_grads)
    _write_u32(f, UInt32(len(parameters.tensors)))
    _write_u32(f, UInt32(num_momentum + num_rms))

    for i in range(len(parameters.tensors)):
        _write_tensor(
            f,
            parameters.tensors.symbols_ref[i],
            OPTIM_KIND_NONE,
            parameters.tensors.data_ref[i],
        )

    for i in range(num_momentum):
        _write_tensor(
            f,
            momentum_grads.symbols_ref[i],
            OPTIM_KIND_MOMENTUM,
            momentum_grads.data_ref[i],
        )

    for i in range(num_rms):
        _write_tensor(
            f, rms_grads.symbols_ref[i], OPTIM_KIND_RMS, rms_grads.data_ref[i]
        )

    f.close()


def load_checkpoint(path: String, mut parameters: Parameters) raises -> CheckpointInfo:
    """
    Restores model tensors from a checkpoint written by
    `save_checkpoint`/`save_checkpoint_with_optim`.

    Args:
        path: Checkpoint file path to read.
        parameters: The model's tensors to restore data into.

    Raises:
        If the file is missing, isn't a basalt checkpoint, or has an
        unsupported version.

    Returns:
        Metadata about what was found in the checkpoint file.

    Notes:
        Tensors are matched by symbol id against the entries already
        allocated in `parameters.tensors` (so the graph that allocated
        them must match the one the checkpoint was saved from). Entries
        for symbol ids not present in `parameters.tensors` are skipped
        rather than erroring, so loading a checkpoint from a slightly
        different (e.g. extended) graph degrades gracefully instead of
        crashing. Any optimizer state present in the file is ignored -
        use `load_checkpoint_with_optim` to restore it.
    """
    var f = open(path, "r")
    var data = f.read_bytes()
    f.close()

    var offset = 0
    var magic = _read_u32(data, offset)
    if magic != CHECKPOINT_MAGIC:
        raise Error("Not a basalt checkpoint file (bad magic): " + path)

    var version = _read_u32(data, offset)
    if version != CHECKPOINT_VERSION:
        raise Error("Unsupported checkpoint version: " + String(version))

    var iter = Int(_read_u32(data, offset))
    var num_tensors = Int(_read_u32(data, offset))
    var num_optim = Int(_read_u32(data, offset))

    for _ in range(num_tensors):
        _read_tensor_into(data, offset, parameters.tensors)

    # Skip any optimizer-state entries present in the file.
    for _ in range(num_optim):
        var header = _read_entry_header(data, offset)
        offset += header[2].num_elements() * size_of[f32]()

    return CheckpointInfo(iter, num_optim > 0)


def load_checkpoint_with_optim(
    path: String,
    mut parameters: Parameters,
    mut momentum_grads: Collection,
    mut rms_grads: Collection,
) raises -> CheckpointInfo:
    """
    Like `load_checkpoint`, but also restores Adam's per-parameter
    momentum/rms state into the given collections if the checkpoint file
    has it.

    Args:
        path: Checkpoint file path to read.
        parameters: The model's tensors to restore data into.
        momentum_grads: `Adam.momentum_grads` to restore data into.
        rms_grads: `Adam.rms_grads` to restore data into.

    Raises:
        If the file is missing, isn't a basalt checkpoint, or has an
        unsupported version.

    Returns:
        Metadata about what was found in the checkpoint file.
    """
    var f = open(path, "r")
    var data = f.read_bytes()
    f.close()

    var offset = 0
    var magic = _read_u32(data, offset)
    if magic != CHECKPOINT_MAGIC:
        raise Error("Not a basalt checkpoint file (bad magic): " + path)

    var version = _read_u32(data, offset)
    if version != CHECKPOINT_VERSION:
        raise Error("Unsupported checkpoint version: " + String(version))

    var iter = Int(_read_u32(data, offset))
    var num_tensors = Int(_read_u32(data, offset))
    var num_optim = Int(_read_u32(data, offset))

    for _ in range(num_tensors):
        _read_tensor_into(data, offset, parameters.tensors)

    for _ in range(num_optim):
        var header = _read_entry_header(data, offset)
        var symbol_id = header[0]
        var kind = header[1]
        var shape = header[2]
        var n_bytes = shape.num_elements() * size_of[f32]()

        if kind == OPTIM_KIND_MOMENTUM:
            var index = momentum_grads.get_index(symbol_id)
            if index == -1:
                offset += n_bytes
                continue
            ref tensor = momentum_grads.data_ref[index]
            var out_ptr = tensor.mut_ptr().bitcast[UInt8]()
            for i in range(n_bytes):
                out_ptr[i] = data[offset + i]
            offset += n_bytes
        else:
            var index = rms_grads.get_index(symbol_id)
            if index == -1:
                offset += n_bytes
                continue
            ref tensor = rms_grads.data_ref[index]
            var out_ptr = tensor.mut_ptr().bitcast[UInt8]()
            for i in range(n_bytes):
                out_ptr[i] = data[offset + i]
            offset += n_bytes

    return CheckpointInfo(iter, num_optim > 0)
