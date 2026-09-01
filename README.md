# ComfyUI T-LoKr for Anima

This uv-managed package supplies a dedicated ComfyUI `TLoKrLoader` node for
the T-LoKr v1 safetensors written by
[`anima-trainer`](https://github.com/rupansh/anima-trainer-sm120). It preserves
the defining behavior that the ordinary LoKr loader cannot: the `W2=A@B`
factorization and the active-rank prefix selected at each flow timestep.

The included `swarmui_extension/TLoKrExtension` adds a matching **T-LoKr
(Anima)** group to SwarmUI's Advanced Options and generates the loader node
between the base model and sampler. It registers this repository as SwarmUI's
`AutoInstall` feature, so self-start ComfyUI backends clone/update the node and
install `requirements.txt` automatically.

The AutoInstall URL is wired to `https://github.com/rupansh/comfy-tlokr`; that
repository must contain this checkout before a fresh SwarmUI instance can clone
it. A local development checkout can use the symlink below immediately.

## Install in the bundled ComfyUI backend

For a development checkout, expose the package to the bundled backend as a
custom node:

```bash
ln -s "$(pwd)" /media/rupansh/wdblack/sd/SwarmUI/dlbackend/ComfyUI/custom_nodes/ComfyUI-TLoKr
cp -a swarmui_extension/TLoKrExtension /media/rupansh/wdblack/sd/SwarmUI/src/Extensions/
```

Then rebuild/restart SwarmUI with `launch-linux-dev.sh` (or its normal update
workflow). On a normal SwarmUI self-start backend, the included extension
instead clones this repository into its managed `DLNodes/comfy-tlokr` directory
automatically. Put the adapter in ComfyUI's `models/loras` folder. For the
supplied example, choose `melted1-tlokr-fp8_e000030.safetensors` and strength
`1.0`.

## Compatibility contract

The input must already be a ComfyUI `MODEL` containing the matching Anima DiT.
The node looks up the checkpoint's exact `diffusion_model.blocks.*` Linear
targets and refuses a different architecture, missing targets, shape changes,
or a non-T-LoKr checkpoint. The Anima model's denoiser timestep must use the
trainer convention: normalized `t` in `[0,1]`, with `t=1` noisy and `t=0`
clean. This node intentionally does not reinterpret arbitrary sigma schedules.

Only T-LoKr v1 is accepted. Its safetensor metadata must include
`anima_adapter_type=tlokr` and `anima_adapter_format=1`; the node never opens
pickle checkpoints.

## Development

`uv` manages the package metadata and lockfile. In this restricted environment,
use a writable temporary uv cache:

```bash
UV_CACHE_DIR=/tmp/comfy-tlokr-uv-cache uv sync
```

The node intentionally gets PyTorch from ComfyUI rather than vendoring a
second torch installation. Run the tests with the backend's Python:

```bash
/media/rupansh/wdblack/sd/SwarmUI/dlbackend/ComfyUI/venv/bin/python \
  -m unittest discover -s tests -v
```
