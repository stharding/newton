"""Minimal GPU test to verify setup works."""

from gpu.host import DeviceContext
from gpu import global_idx
from layout import Layout, LayoutTensor
from sys import has_accelerator


fn simple_kernel[size: Int, layout: Layout](
    output: LayoutTensor[DType.float64, layout, MutAnyOrigin],
):
    """Simple kernel: each thread writes its index."""
    var idx = Int(global_idx.x)
    if idx < size:
        output[idx] = Float64(idx)


fn main() raises:
    if not has_accelerator():
        print("No GPU found!")
        return

    print("GPU test starting...")

    var ctx = DeviceContext()
    print("Using GPU: " + ctx.name())

    comptime size = 16
    comptime layout = Layout.row_major(size)

    var device_buf = ctx.enqueue_create_buffer[DType.float64](size)
    var host_buf = ctx.enqueue_create_host_buffer[DType.float64](size)
    ctx.synchronize()

    var tensor = LayoutTensor[DType.float64, layout](device_buf)

    ctx.enqueue_function[simple_kernel[size, layout], simple_kernel[size, layout]](
        tensor,
        grid_dim=1,
        block_dim=size,
    )

    ctx.enqueue_copy(host_buf, device_buf)
    ctx.synchronize()

    print("Results:")
    for i in range(size):
        print("  [" + String(i) + "] = " + String(host_buf[i]))
