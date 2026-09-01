using Newtonsoft.Json.Linq;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace ComfyTLoKr.Swarm;

/// <summary>
/// Routes T-LoKr entries selected in SwarmUI's normal LoRA selector to the
/// timestep-aware ComfyUI loader.
/// </summary>
public class TLoKrExtension : Extension
{
    private static T2IModel FindLoraModel(string configuredName)
    {
        if (string.IsNullOrWhiteSpace(configuredName))
        {
            return null;
        }
        T2IModelHandler handler = Program.T2IModelSets["LoRA"];
        string normalized = configuredName.Replace('\\', '/');
        string withoutExtension = normalized.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)
            ? normalized[..^".safetensors".Length]
            : normalized;
        string[] candidates =
        [
            configuredName,
            normalized,
            normalized + ".safetensors",
            withoutExtension,
            withoutExtension + ".safetensors"
        ];
        foreach (string candidate in candidates)
        {
            if (handler.Models.TryGetValue(candidate, out T2IModel model))
            {
                return model;
            }
        }
        return handler.Models.Values.FirstOrDefault(model =>
            string.Equals(model.Name.Replace('\\', '/'), normalized, StringComparison.OrdinalIgnoreCase)
            || string.Equals(model.Name.Replace('\\', '/'), withoutExtension, StringComparison.OrdinalIgnoreCase)
            || string.Equals(model.Name.Replace('\\', '/'), withoutExtension + ".safetensors", StringComparison.OrdinalIgnoreCase));
    }

    private static bool IsTLoKr(T2IModel model)
    {
        if (model?.RawFilePath is null)
        {
            return false;
        }
        try
        {
            JObject header = T2IModel.GetMetadataHeaderFrom(model.RawFilePath);
            JObject metadata = header?["__metadata__"] as JObject;
            return string.Equals(metadata?.Value<string>("anima_adapter_type"), "tlokr", StringComparison.OrdinalIgnoreCase);
        }
        catch (Exception ex)
        {
            Logs.Debug($"Unable to inspect LoRA metadata for '{model.Name}': {ex.Message}");
            return false;
        }
    }

    private static void RouteTLoKrLoras(WorkflowGenerator g)
    {
        foreach (JProperty property in g.Workflow.Properties().ToList())
        {
            if (property.Value is not JObject node)
            {
                continue;
            }
            string classType = node.Value<string>("class_type");
            bool hasClip = classType == "LoraLoader";
            if (!hasClip && classType != "LoraLoaderModelOnly")
            {
                continue;
            }
            if (node["inputs"] is not JObject inputs)
            {
                continue;
            }
            string configuredName = inputs.Value<string>("lora_name");
            T2IModel lora = FindLoraModel(configuredName);
            if (!IsTLoKr(lora))
            {
                continue;
            }
            if (!g.Features.Contains("tlokr"))
            {
                throw new SwarmUserErrorException(
                    "A T-LoKr adapter was selected in the LoRA list, but the ComfyUI T-LoKr node is not installed.");
            }

            // Preserve the node's position and IDs so mixed ordinary LoRAs and
            // T-LoKr adapters retain exactly the order selected by the user.
            inputs["tlokr_name"] = inputs["lora_name"];
            inputs.Remove("lora_name");
            if (hasClip)
            {
                inputs.Remove("strength_clip");
                node["class_type"] = "TLoKrLoaderWithClip";
            }
            else
            {
                node["class_type"] = "TLoKrLoader";
            }
        }
    }

    public override void OnInit()
    {
        // Self-start backends clone this repository on startup and keep it
        // current. Swarm installs requirements.txt with the backend Python.
        InstallableFeatures.RegisterInstallableFeature(new(
            "ComfyUI T-LoKr (Anima)",
            "tlokr",
            "https://github.com/rupansh/comfy-tlokr",
            "rupansh",
            "Installs the ComfyUI T-LoKr loader for Anima adapters from rupansh/comfy-tlokr.",
            AutoInstall: true
        ));
        ComfyUIBackendExtension.NodeToFeatureMap["TLoKrLoader"] = "tlokr";
        ComfyUIBackendExtension.NodeToFeatureMap["TLoKrLoaderWithClip"] = "tlokr";

        // Core SwarmUI creates standard LoraLoader nodes at -10. Rewrite only
        // the T-LoKr entries immediately afterward; ordinary LoRAs are kept.
        WorkflowGenerator.AddModelGenStep(RouteTLoKrLoras, -9.5);
    }
}
