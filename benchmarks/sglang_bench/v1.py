"""Four-forward V1 contract over SGLang's existing native H3 implementations."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    FastH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import FastH3Pipeline


@dataclass
class FastH3V1SamplingParams(MiniMaxH3SamplingParams):
    num_inference_steps: int = 4
    quality: str = "lossless"

    def _validate(self):
        super()._validate()
        if self.num_inference_steps != 4:
            raise ValueError(
                "FastH3 V1 requires exactly four DiT forwards / five sigma points"
            )
        if self.task is not None and self.task.strip().lower() != "t2va":
            raise ValueError("FastH3 V1 baseline serves t2va only")


@dataclass
class FastH3V1PipelineConfig(FastH3PipelineConfig):
    vsa_sparsity: float = 0.9


class FastH3V1Pipeline(FastH3Pipeline):
    pipeline_name = "FastH3V1BaselinePipeline"
    pipeline_config_cls = FastH3V1PipelineConfig
    sampling_params_cls = FastH3V1SamplingParams
