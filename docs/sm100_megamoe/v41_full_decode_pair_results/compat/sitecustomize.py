"""CUTLASS DSL 4.8.1 compatibility alias for vLLM's tcgen05 MXFP8 call site.

The 4.8.1 binding splits the block-scale selectors out of the generated
``nvvm`` namespace.  Its own experimental wrapper retains the older
``Tcgen05MMAScaleVecSize`` spelling and documents it as a compatibility
namespace, so alias that exact wrapper object before vLLM imports tcgen05.
"""

try:
    import inspect
    from cutlass._mlir.dialects import nvvm as _nvvm
    from cutlass.experimental.primitives import nvvm_wrapper as _nvvm_wrapper

    if not hasattr(_nvvm, "Tcgen05MMAScaleVecSize"):
        _nvvm.Tcgen05MMAScaleVecSize = _nvvm_wrapper.Tcgen05MMAScaleVecSize

    _raw_mma_block_scale = getattr(_nvvm, "tcgen05_mma_block_scale", None)
    if _raw_mma_block_scale is not None:
        _parameters = inspect.signature(_raw_mma_block_scale).parameters
        if "scale_vec_size" not in _parameters and "block_scale" in _parameters:
            def _compat_mma_block_scale(*args, scale_vec_size=None, **kwargs):
                if scale_vec_size is not None:
                    if "block_scale" in kwargs:
                        raise TypeError("block_scale and scale_vec_size are mutually exclusive")
                    kwargs["block_scale"] = _nvvm_wrapper._to_tcgen05_block_scale(scale_vec_size)
                return _raw_mma_block_scale(*args, **kwargs)
            _compat_mma_block_scale.__name__ = "tcgen05_mma_block_scale"
            _compat_mma_block_scale.__doc__ = _raw_mma_block_scale.__doc__
            _nvvm.tcgen05_mma_block_scale = _compat_mma_block_scale
except ImportError:
    pass
