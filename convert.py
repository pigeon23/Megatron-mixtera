from megatron.bridge import AutoBridge

if AutoBridge.can_handle("LlamaForCausalLM"):
    print("✅ Model is supported!")
    bridge = AutoBridge.from_hf_pretrained("LlamaForCausalLM")
else:
    print("❌ Model requires a custom bridge implementation")