"""lm_client — a researcher-first batching client built on LiteLLM.

`lm_client` wraps [LiteLLM](https://docs.litellm.ai) with the workflow pieces
that tend to show up once an experiment stops being "one request in, one string
out": concurrent batch execution, notebook-safe sync calls, partial-failure
handling, client-side throttling, and optional multi-replica routing. Public
results are typed in both synchronous and asynchronous Python. Multimodal (VLM)
inputs are supported via the standard OpenAI content-block format; use
[image_block][lm_client.image_block] to encode local image files or raw bytes
before sending. It supports two operating modes:

**Single-endpoint mode** — one model, one server:

```python
from lm_client import LMClient

with LMClient(
    model="openai/gpt-4o-mini", api_base="http://localhost:8000/v1"
) as client:
    result = client.generate("What is 2 + 2?")
    print(result.output_text)  # "4"
```

**Router mode** — multiple replicas with automatic load-balancing:

```python
from lm_client import LMClient, DeploymentConfig

client = LMClient(
    deployments={
        "gpu-0": DeploymentConfig(
            model="hosted_vllm/meta-llama/Meta-Llama-3-8B-Instruct",
            api_base="http://gpu0:8000/v1",
        ),
        "gpu-1": DeploymentConfig(
            model="hosted_vllm/meta-llama/Meta-Llama-3-8B-Instruct",
            api_base="http://gpu1:8000/v1",
        ),
    }
)
```

**Rate limiting** — pass `rpm` and/or `tpm` to enable automatic throttling:

```python
client = LMClient(
    model="openai/gpt-4o",
    api_base="https://api.openai.com/v1",
    rpm=500,
    tpm=100_000,
)
```

When targeting hosted providers, export the relevant provider environment
variable (for example `OPENAI_API_KEY`) before constructing the client.
Advanced library integrations can still pass `api_key` directly when the
secret comes from a secret manager or another in-process credential source.

If you only need a small number of single requests, plain LiteLLM or the
provider SDK is usually simpler. `lm_client` is most useful when you want to
push larger workloads through a notebook, script, or local inference stack.

**Async usage** — every public method has an `a`-prefixed async counterpart:

```python
import asyncio
from lm_client import LMClient


async def main() -> None:
    async with LMClient(...) as client:
        results = await client.agenerate_batch(["prompt 1", "prompt 2"])
        for r in results:
            if r is not None:
                print(r.output_text)


asyncio.run(main())
```

**Public symbols:**

- [LMClient][lm_client.LMClient] — main client; generation, embedding, and
  transcription in both sync and async forms, with optional rate limiting and
  router mode.
- [DeploymentConfig][lm_client.DeploymentConfig] — per-replica configuration
  used in router mode.
- [BatchResult][lm_client.BatchResult] — generic container returned by
  `*_batch` methods.
- [GenerationResult][lm_client.GenerationResult] — typed result from a
  text-generation call.
- [EmbeddingResult][lm_client.EmbeddingResult] — typed result from an
  embedding call.
- [TranscriptionResult][lm_client.TranscriptionResult] — typed result from an
  audio-transcription call.
- [RateLimiter][lm_client.RateLimiter] — async token-bucket rate limiter;
  created automatically by [LMClient][lm_client.LMClient] when `rpm` / `tpm`
  are supplied, but can also be used standalone.
- [RateLimiterAcquisitionHandle][lm_client.RateLimiterAcquisitionHandle] —
  opaque handle returned by [acquire][lm_client.RateLimiter.acquire]; passed
  back to [adjust][lm_client.RateLimiter.adjust] after the request completes.
- [TokenUsage][lm_client.TokenUsage] — token-count breakdown attached to
  generation and embedding results.
- [RequestMetrics][lm_client.RequestMetrics] — per-request timing and routing
  metadata.
- [ToolCall][lm_client.ToolCall] — a structured tool-call emitted by a model
  during generation.
- [image_block][lm_client.image_block] — build an image content block from a
  local file, raw bytes, or URL for multimodal (VLM) chat messages.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lm-client")
except PackageNotFoundError:
    __version__ = "unknown"

from lm_client.client import LMClient
from lm_client.rate_limiter import RateLimiter, RateLimiterAcquisitionHandle
from lm_client.types import (
    BatchResult,
    DeploymentConfig,
    EmbeddingResult,
    GenerationResult,
    RequestMetrics,
    TokenUsage,
    ToolCall,
    TranscriptionResult,
    image_block,
)

__all__ = [
    "__version__",
    "BatchResult",
    "DeploymentConfig",
    "EmbeddingResult",
    "GenerationResult",
    "image_block",
    "LMClient",
    "RateLimiter",
    "RateLimiterAcquisitionHandle",
    "RequestMetrics",
    "TokenUsage",
    "ToolCall",
    "TranscriptionResult",
]
