# T-LoKr SwarmUI extension

This extension integrates with SwarmUI's existing **LoRA** selector. When a
selected adapter's safetensor metadata declares `anima_adapter_type=tlokr`,
the generated `LoraLoader` node is replaced with the timestep-aware T-LoKr
loader; ordinary LoRAs are left unchanged. On a self-start ComfyUI backend it
automatically clones and updates
`https://github.com/rupansh/comfy-tlokr`, then installs its `requirements.txt`.
The repository must be populated with this project before a fresh backend can
clone it; local development can install the checkout directly as a custom node.

It becomes available only when the ComfyUI backend reports the T-LoKr nodes.
Enter the adapter through the normal LoRA selector, for example
`melted1-tlokr-fp8_e000030.safetensors`, and use LoRA weight `1.0` for the
trainer's native scale.
