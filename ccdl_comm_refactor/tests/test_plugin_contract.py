from ccdl_comm.config import CompressionConfig
from ccdl_comm.plugin import CCDLCommunicationPlugin


def test_plugin_exposes_stable_name_and_explanation():
    plugin = CCDLCommunicationPlugin()

    assert plugin.name == "ccdl"
    assert "native DDP" in " ".join(plugin.explain(CompressionConfig()))


def test_plugin_rejects_non_native_ddp_context():
    plugin = CCDLCommunicationPlugin()
    context = {"training_backend": "fsdp", "device_type": "cuda"}

    decision = plugin.plan(context, CompressionConfig())

    assert decision.enabled is False
    assert decision.fallback == "bf16_compress"
    assert "native_ddp" in decision.reason
