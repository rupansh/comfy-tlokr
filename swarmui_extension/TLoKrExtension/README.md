# T-LoKr SwarmUI extension

This extension adds the **T-LoKr (Anima)** advanced group and inserts a
`TLoKrLoader` directly after SwarmUI's selected base model. On a self-start
ComfyUI backend it automatically clones and updates
`https://github.com/rupansh/comfy-tlokr`, then installs its `requirements.txt`.
The repository must be populated with this project before a fresh backend can
clone it; local development can install the checkout directly as a custom node.

It becomes available only when the ComfyUI backend reports the `TLoKrLoader`
node. Enter the adapter filename relative to ComfyUI's `models/loras` folder,
for example `melted1-tlokr-fp8_e000030.safetensors`, and use strength `1.0` for
the trainer's native scale.
