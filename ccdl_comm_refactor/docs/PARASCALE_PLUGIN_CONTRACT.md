# ParaScale Communication Plugin Contract Draft

This is the first contract draft for using CCDL from ParaScale.

## Context from ParaScale

ParaScale should pass a small context object to the plugin:

```python
context = {
    "training_backend": "native_ddp",
    "device_type": "cuda",
    "world_size": 2,
    "torch_version": "2.4.0",
}
```

## CCDL config

```python
from ccdl_comm import CompressionConfig

config = CompressionConfig(
    bit=8,
    group_size=64,
    topk=0,
    quant_type="linear",
    error_feedback=True,
)
```

## Planning

```python
from ccdl_comm import CCDLCommunicationPlugin

plugin = CCDLCommunicationPlugin()
decision = plugin.plan(context, config)
```

The first implementation only enables CCDL when:

```text
training_backend == "native_ddp"
device_type == "cuda"
```

Otherwise the plugin returns a disabled decision with a fallback such as
`bf16_compress`.

## Future hook API

Planned API:

```python
hook, state = plugin.build_ddp_hook(config)
model.register_comm_hook(state, hook)
```

The hook must return:

```text
torch.futures.Future[Tensor]
```

This is not implemented in the initial scaffolding milestone.
