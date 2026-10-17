"""Opt-in native registration and measurement hooks; no inference replacements."""

import os

_registered = False


def register():
    global _registered
    # Entry points are auto-discovered by native SGLang; ordinary commands stay untouched.
    if "sglang_bench" not in os.environ.get("SGLANG_PLUGINS", "").split(","):
        return
    if _registered:
        return
    from sglang.multimodal_gen.registry import register_pipeline

    from .v1 import FastH3V1Pipeline, FastH3V1PipelineConfig, FastH3V1SamplingParams

    register_pipeline(
        FastH3V1Pipeline,
        sampling_param_cls=FastH3V1SamplingParams,
        pipeline_config_cls=FastH3V1PipelineConfig,
        hf_model_paths=["FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"],
        overwrite=True,
    )
    from sglang.multimodal_gen.configs.pipeline_configs.wan import (
        FastWan2_1_T2V_480P_Config,
    )
    from sglang.multimodal_gen.configs.sample.wan import FastWanT2V480PConfig
    from sglang.multimodal_gen.runtime.pipelines.wan_dmd_pipeline import WanDMDPipeline

    register_pipeline(
        WanDMDPipeline,
        sampling_param_cls=FastWanT2V480PConfig,
        pipeline_config_cls=FastWan2_1_T2V_480P_Config,
        hf_model_paths=["FastVideo/FastWan2.1-T2V-14B-Diffusers"],
        overwrite=True,
    )
    from .observer import install_hooks

    install_hooks()
    _registered = True
