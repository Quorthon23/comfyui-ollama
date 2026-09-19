import { app } from "/scripts/app.js";

app.registerExtension({
  name: "Comfy.OllamaNode",
  aboutPageBadges: [
    {
      label: "ComfyUI-Ollama",
      url: "https://github.com/stavsap/comfyui-ollama",
      icon: "pi pi-github",
    },
  ],
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (["OllamaGenerate", "OllamaGenerateAdvance", "OllamaVision", "OllamaConnectivityV2", "UnslothConnectivity"].includes(nodeData.name)) {
      const originalNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = async function () {
        if (originalNodeCreated) {
          originalNodeCreated.apply(this, arguments);
        }

        const isUnsloth = nodeData.name === "UnslothConnectivity";
        const urlWidget = this.widgets.find((w) => w.name === "url");
        const apiKeyWidget = this.widgets.find((w) => w.name === "api_key");
        const modelWidget = this.widgets.find((w) => w.name === "model");
        const quantWidget = this.widgets.find((w) => w.name === "quantization");
        const ctxWidget = this.widgets.find((w) => w.name === "context_length");
        let refreshButtonWidget = {};
        let loadModelButtonWidget = null;
        let unloadModelButtonWidget = null;
        if (nodeData.name === "OllamaConnectivityV2" || isUnsloth) {
          refreshButtonWidget = this.addWidget("button", "🔄 Reconnect");
        }
        if (isUnsloth) {
          loadModelButtonWidget = this.addWidget("button", "⚡ Load Model");
          unloadModelButtonWidget = this.addWidget("button", "🗑️ Unload Model");
        }

        const fetchModels = async () => {
          if (isUnsloth) {
            const response = await fetch("/unsloth/get_models", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                url: urlWidget ? urlWidget.value : "http://127.0.0.1:8888",
                api_key: apiKeyWidget ? apiKeyWidget.value : "",
              }),
            });

            if (response.ok) {
              const models = await response.json();
              console.debug("Fetched Unsloth models:", models);
              return models;
            } else {
              throw new Error(`HTTP ${response.status}`);
            }
          } else {
            const response = await fetch("/ollama/get_models", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                url: urlWidget ? urlWidget.value : "http://127.0.0.1:11434",
              }),
            });

            if (response.ok) {
              const models = await response.json();
              console.debug("Fetched Ollama models:", models);
              return models;
            } else {
              throw new Error(response);
            }
          }
        };

        const updateVariants = async (targetModel) => {
          if (!isUnsloth || !quantWidget) return;
          const currentModel = targetModel !== undefined ? targetModel : (modelWidget ? modelWidget.value : "");
          if (!currentModel) {
            quantWidget.options.values = ["(none)"];
            quantWidget.value = "(none)";
            this.setDirtyCanvas(true);
            return;
          }

          try {
            const response = await fetch("/unsloth/get_variants", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                url: urlWidget ? urlWidget.value : "http://127.0.0.1:8888",
                api_key: apiKeyWidget ? apiKeyWidget.value : "",
                model: currentModel,
              }),
            });

            if (response.ok) {
              const variants = await response.json();
              console.debug("Fetched downloaded Unsloth variants for", currentModel, variants);
              const prevVal = quantWidget.value;

              if (Array.isArray(variants) && variants.length > 0) {
                quantWidget.options.values = variants;
                if (variants.includes(prevVal)) {
                  quantWidget.value = prevVal;
                } else {
                  quantWidget.value = variants[0];
                }
              } else {
                // Model is not downloaded or has no downloaded GGUF quantizations
                quantWidget.options.values = ["(none)"];
                quantWidget.value = "(none)";
              }
            }
          } catch (err) {
            console.warn("Could not fetch variants for model:", err);
          }
          this.setDirtyCanvas(true);
        };

        const loadModel = async () => {
          if (!modelWidget || !modelWidget.value) {
            app.extensionManager.toast.add({
              severity: "warn",
              summary: "Unsloth",
              detail: "Please select a model first",
              life: 3000,
            });
            return;
          }

          if (loadModelButtonWidget) {
            loadModelButtonWidget.name = "⏳ Loading...";
            this.setDirtyCanvas(true);
          }

          const selectedQuant = quantWidget && quantWidget.value && quantWidget.value !== "(none)" && quantWidget.value !== "default"
            ? quantWidget.value
            : "";
          const selectedCtx = ctxWidget && ctxWidget.value ? parseInt(ctxWidget.value, 10) : 32768;

          try {
            const response = await fetch("/unsloth/load_model", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                url: urlWidget ? urlWidget.value : "http://127.0.0.1:8888",
                api_key: apiKeyWidget ? apiKeyWidget.value : "",
                model: modelWidget.value,
                quantization: selectedQuant,
                context_length: selectedCtx,
              }),
            });

            if (response.ok) {
              if (loadModelButtonWidget) loadModelButtonWidget.name = "✅ Loaded";
              const quantLabel = selectedQuant ? ` (${selectedQuant})` : "";
              const ctxLabel = selectedCtx ? ` [${Math.round(selectedCtx / 1024)}K]` : "";
              app.extensionManager.toast.add({
                severity: "success",
                summary: "Unsloth",
                detail: `Model '${modelWidget.value}'${quantLabel}${ctxLabel} loaded into memory!`,
                life: 4000,
              });
              setTimeout(() => {
                if (loadModelButtonWidget) loadModelButtonWidget.name = "⚡ Load Model";
                this.setDirtyCanvas(true);
              }, 3000);
            } else {
              const err = await response.json();
              throw new Error(err.error || `HTTP ${response.status}`);
            }
          } catch (error) {
            console.error("Error loading model:", error);
            if (loadModelButtonWidget) loadModelButtonWidget.name = "❌ Failed";
            app.extensionManager.toast.add({
              severity: "error",
              summary: "Unsloth Load Error",
              detail: error.message || "Failed to load model into memory",
              life: 5000,
            });
            setTimeout(() => {
              if (loadModelButtonWidget) loadModelButtonWidget.name = "⚡ Load Model";
              this.setDirtyCanvas(true);
            }, 3000);
          }
          this.setDirtyCanvas(true);
        };

        const unloadModel = async () => {
          if (!modelWidget || !modelWidget.value) {
            app.extensionManager.toast.add({
              severity: "warn",
              summary: "Unsloth",
              detail: "Please select a model first",
              life: 3000,
            });
            return;
          }

          if (unloadModelButtonWidget) {
            unloadModelButtonWidget.name = "⏳ Unloading...";
            this.setDirtyCanvas(true);
          }

          try {
            const response = await fetch("/unsloth/unload_model", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                url: urlWidget ? urlWidget.value : "http://127.0.0.1:8888",
                api_key: apiKeyWidget ? apiKeyWidget.value : "",
                model: modelWidget.value,
              }),
            });

            if (response.ok) {
              if (unloadModelButtonWidget) unloadModelButtonWidget.name = "✅ Unloaded";
              app.extensionManager.toast.add({
                severity: "success",
                summary: "Unsloth",
                detail: `Model '${modelWidget.value}' unloaded from memory!`,
                life: 4000,
              });
              setTimeout(() => {
                if (unloadModelButtonWidget) unloadModelButtonWidget.name = "🗑️ Unload Model";
                this.setDirtyCanvas(true);
              }, 3000);
            } else {
              const err = await response.json();
              throw new Error(err.error || `HTTP ${response.status}`);
            }
          } catch (error) {
            console.error("Error unloading model:", error);
            if (unloadModelButtonWidget) unloadModelButtonWidget.name = "❌ Failed";
            app.extensionManager.toast.add({
              severity: "error",
              summary: "Unsloth Unload Error",
              detail: error.message || "Failed to unload model from memory",
              life: 5000,
            });
            setTimeout(() => {
              if (unloadModelButtonWidget) unloadModelButtonWidget.name = "🗑️ Unload Model";
              this.setDirtyCanvas(true);
            }, 3000);
          }
          this.setDirtyCanvas(true);
        };

        const updateModels = async () => {
          if (refreshButtonWidget) refreshButtonWidget.name = "⏳ Fetching...";

          let models = [];
          try {
            models = await fetchModels();
          } catch (error) {
            console.error("Error fetching models:", error);
            app.extensionManager.toast.add({
              severity: "error",
              summary: isUnsloth ? "Unsloth connection error" : "Ollama connection error",
              detail: isUnsloth
                ? "Make sure Unsloth server is running and API key is valid"
                : "Make sure Ollama server is running",
              life: 5000,
            });
            if (refreshButtonWidget) refreshButtonWidget.name = "🔄 Reconnect";
            this.setDirtyCanvas(true);
            return;
          }

          const prevValue = modelWidget.value;

          // Update modelWidget options and value
          modelWidget.options.values = models;
          console.debug("Updated modelWidget.options.values:", modelWidget.options.values);

          if (models.includes(prevValue)) {
            modelWidget.value = prevValue; // stay on current.
          } else if (models.length > 0) {
            modelWidget.value = models[0]; // set first as default.
          }

          if (refreshButtonWidget) refreshButtonWidget.name = "🔄 Reconnect";
          this.setDirtyCanvas(true);
          console.debug("Updated modelWidget.value:", modelWidget.value);

          await updateVariants(modelWidget.value);
        };

        if (urlWidget) urlWidget.callback = updateModels;
        if (apiKeyWidget) apiKeyWidget.callback = updateModels;
        if (modelWidget) {
          const origModelCb = modelWidget.callback;
          modelWidget.callback = async function (v) {
            if (origModelCb) origModelCb.apply(this, arguments);
            await updateVariants(v);
          };
        }
        if (refreshButtonWidget) refreshButtonWidget.callback = updateModels;
        if (loadModelButtonWidget) loadModelButtonWidget.callback = loadModel;
        if (unloadModelButtonWidget) unloadModelButtonWidget.callback = unloadModel;

        const dummy = async () => {
          // calling async method will update the widgets with actual value from the browser and not the default from Node definition.
        };

        // Initial update
        await dummy(); // this will cause the widgets to obtain the actual value from web page.
        await updateModels();
        await updateVariants(modelWidget ? modelWidget.value : "");
      };
    }
  },
});

