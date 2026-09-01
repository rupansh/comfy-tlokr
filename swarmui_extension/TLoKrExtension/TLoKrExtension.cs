using Newtonsoft.Json.Linq;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace ComfyTLoKr.Swarm;

/// <summary>Adds a T-LoKr selector to SwarmUI's Comfy workflow generator.</summary>
public class TLoKrExtension : Extension
{
    public static T2IRegisteredParam<string> TLoKr;

    public static T2IRegisteredParam<double> TLoKrStrength;

    public static T2IParamGroup TLoKrGroup;

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
        // ComfyUI marks the capability available only after it sees the node.
        ComfyUIBackendExtension.NodeToFeatureMap["TLoKrLoader"] = "tlokr";
        TLoKrGroup = new("T-LoKr (Anima)", Toggles: true, Open: false, IsAdvanced: true);
        TLoKr = T2IParamTypes.Register<string>(new(
            "T-LoKr Adapter",
            "[T-LoKr]\nA T-LoKr v1 safetensor in the ComfyUI LoRA model folder. " +
            "It must match the selected Anima DiT base model.",
            "",
            IgnoreIf: "",
            GetValues: session => Program.T2IModelSets["LoRA"].ListModelNamesFor(session),
            Group: TLoKrGroup,
            FeatureFlag: "tlokr",
            Subtype: "LoRA",
            OrderPriority: 1
        ));
        TLoKrStrength = T2IParamTypes.Register<double>(new(
            "T-LoKr Strength",
            "[T-LoKr]\nAdapter multiplier. 1.0 reproduces the trained scale; negative values invert it.",
            "1",
            Min: -100,
            Max: 100,
            Step: 0.01,
            Group: TLoKrGroup,
            FeatureFlag: "tlokr",
            OrderPriority: 2
        ));

        WorkflowGenerator.AddStep(g =>
        {
            if (!g.UserInput.TryGet(TLoKr, out string tlokrName) || string.IsNullOrWhiteSpace(tlokrName))
            {
                return;
            }
            if (!g.Features.Contains("tlokr"))
            {
                throw new SwarmUserErrorException("T-LoKr was selected, but the ComfyUI TLoKrLoader node is not installed.");
            }
            T2IModelHandler loraHandler = Program.T2IModelSets["LoRA"];
            if (!loraHandler.Models.TryGetValue(tlokrName + ".safetensors", out T2IModel tlokr)
                && !loraHandler.Models.TryGetValue(tlokrName, out tlokr))
            {
                throw new SwarmUserErrorException($"T-LoKr adapter '{tlokrName}' was not found in the LoRA model folder.");
            }
            g.FinalLoadedModelList.Add(tlokr);
            string node = g.CreateNode("TLoKrLoader", new JObject()
            {
                ["model"] = g.CurrentModel.Path,
                ["tlokr_name"] = tlokr.ToString(g.ModelFolderFormat),
                ["strength_model"] = g.UserInput.Get(TLoKrStrength, 1.0),
            });
            g.CurrentModel = g.CurrentModel.WithPath([node, 0]);
        }, -5.5);
    }
}
