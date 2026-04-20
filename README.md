# infermesh

`infermesh` is for researchers and engineers who need to run large LLM jobs
from notebooks, scripts, or local inference stacks without rebuilding the same
concurrency and quota-control layer each time.

It sits on top of LiteLLM and focuses on the parts that usually show up once an
experiment becomes real work:

- concurrent batch generation with ordered results
- notebook-safe sync APIs
- per-item failure handling for long runs
- crash-resilient batches with incremental writes and `--resume` support
- automatic retries with exponential backoff for transient errors
- client-side RPM and TPM throttling
- typed results with token usage and timing metadata
- multi-replica routing for local or clustered inference endpoints

If you only need a handful of one-off requests, use the provider SDK or plain
LiteLLM. `infermesh` earns its keep when throughput control and batch ergonomics
matter more than raw minimalism.

## Install

Python `3.12+` is required.

```bash
python -m pip install infermesh
```

If you use `uv`:

```bash
uv add infermesh
```

Contributor setup, editable installs, and clone-based workflows live in
[CONTRIBUTING.md](https://github.com/VectorInstitute/infermesh/blob/main/CONTRIBUTING.md).

## Quick Start

Set the provider key in your environment first:

```bash
export OPENAI_API_KEY=sk-...
```

The core workflow is "run a batch, keep the results you want, inspect the failures,
and retry only what broke":

```python
from infermesh import LMClient

prompts = [
    "Summarize section 1 in two bullet points.",
    "Summarize section 2 in two bullet points.",
    "Summarize section 3 in two bullet points.",
]

with LMClient(
    model="openai/gpt-4.1-mini",
    max_parallel_requests=32,
    rpm=500,
    tpm=100_000,
) as client:
    batch = client.generate_batch(prompts)

retry_prompts: list[str] = []

for i, result in enumerate(batch):
    if result is None:
        print(f"FAILED: {prompts[i]}\n  {batch.errors[i]}")
        retry_prompts.append(prompts[i])
    else:
        print(result.output_text)
        if result.token_usage is not None:
            print("tokens:", result.token_usage.total_tokens)

if retry_prompts:
    with LMClient(model="openai/gpt-4.1-mini") as retry_client:
        retry_batch = retry_client.generate_batch(retry_prompts)
```

One failing request does not abort the whole batch. Failed items are `None` in
`batch.results`; the exception is in `batch.errors[i]`. This is deliberate: a single
provider error should not wipe out a long experiment.

For large Python batches, set `max_parallel_requests` explicitly. `generate_batch`
and `transcribe_batch` both use a bounded in-flight window when it is set; when it
is unset, they start one coroutine per item up front, which can cause memory pressure
for very large inputs. `embed_batch` is always micro-batched regardless of
`max_parallel_requests` — pass `micro_batch_size` to tune chunk size instead.

This code works in Jupyter notebooks without any `asyncio` setup. The sync API runs a
background event loop so you do not have to.

For a single one-off request:

```python
with LMClient(model="openai/gpt-4.1-mini") as client:
    result = client.generate("What is the capital of France?")
    print(result.output_text)
```

The `model` string follows LiteLLM's `provider/model-name` format. See the
[LiteLLM model list](https://docs.litellm.ai/docs/providers) for all supported
providers:

| Provider | Example |
| --- | --- |
| OpenAI | `"openai/gpt-4.1-mini"` |
| Anthropic | `"anthropic/claude-3-5-sonnet-20241022"` |
| Local vLLM | `"hosted_vllm/meta-llama/Meta-Llama-3-8B-Instruct"` |

`api_base` is optional for hosted providers — LiteLLM already knows their endpoints.
Set it explicitly for local servers or custom deployments. Keep provider secrets in
environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`); local servers that
require no auth work without an `api_key`.

## Generate Text

```python
result = client.generate("Say hello in one sentence.")

print(result.output_text)     # generated text
print(result.token_usage)     # prompt / completion / total token counts
print(result.finish_reason)   # "stop", "length", …
print(result.request_id)      # provider-assigned ID for debugging
```

## Create Embeddings

```python
# Single string → EmbeddingResult
result = client.embed("The quick brown fox")
print(result.embedding)          # list[float]

# Multiple strings → processed in resilient micro-batches by default
batch = client.embed_batch(
    ["sentence one", "sentence two", "sentence three"],
    micro_batch_size=32,
)
vectors = [r.embedding for r in batch if r is not None]
```

## Transcribe Audio

```python
result = client.transcribe("recording.wav")   # path, bytes, or file-like object
print(result.text)
print(result.language)     # detected language code, e.g. "en"
print(result.duration_s)   # audio length in seconds

batch = client.transcribe_batch(["recording-a.wav", "recording-b.wav"])
texts = [r.text if r is not None else None for r in batch]
```

Audio inputs larger than 25 MB are rejected by default. Pass
`max_transcription_bytes=None` only in trusted environments where the server is
expected to accept larger uploads. Disabling the guard means the client may
read and send very large audio files in full.

## CLI

```bash
# Set your key first (or use --env-file .env)
export OPENAI_API_KEY=sk-...

# Generate — single prompt
infermesh generate \
  --model openai/gpt-4.1-mini \
  --api-base https://api.openai.com/v1 \
  --prompt "Hello"

# Generate — from a JSONL file, results to another JSONL file
# Each input line: {"prompt": "..."} or {"messages": [...]} or {"responses_input": "..."}
# Output includes an _index field; a checkpoint file results.checkpoint.sqlite is kept.
infermesh generate \
  --model openai/gpt-4.1-mini \
  --api-base https://api.openai.com/v1 \
  --input-jsonl prompts.jsonl \
  --output-jsonl results.jsonl

# Generate uses a rolling in-flight window by default (128 rows unless you
# override it with --max-parallel-requests): each settled row immediately
# admits the next pending row.

# Resume an interrupted run — reads results.checkpoint.sqlite, skips settled rows, appends the rest
infermesh generate \
  --model openai/gpt-4.1-mini \
  --api-base https://api.openai.com/v1 \
  --input-jsonl prompts.jsonl \
  --output-jsonl results.jsonl \
  --resume

# Resume is strict: the checkpoint file must already exist for this output file
# Duplicate rows are tracked independently, and resumed rows keep their original _index values
# Reordering the input file is safe, but removing or deduplicating rows is not
# Input and output files must be different paths
# Mapper-backed resume is tied to the original mapper implementation

# Custom mapper — transform raw source records before sending to the model
# The mapper receives each record as a dict; must return {"input": ..., "metadata": ...}
infermesh generate \
  --model openai/gpt-4.1-mini \
  --input-jsonl dataset.jsonl \
  --output-jsonl results.jsonl \
  --mapper mypackage.prompts:build_prompt

# Create embeddings
infermesh embed \
  --model text-embedding-3-small \
  --api-base https://api.openai.com/v1 \
  --text "hello world"

# Transcribe audio
infermesh transcribe --model whisper-1 \
  --api-base https://api.openai.com/v1 \
  recording.wav
```

## Advanced Usage

<details>
<summary>Crash-resilient batches (on_result)</summary>

For long runs, pass `on_result` to write each result to disk as it arrives.
A crash or interruption only loses the requests that were in-flight at that
moment — everything already completed is safe on disk.

`generate_batch`, `embed_batch`, and `transcribe_batch` all support the same
per-item callback contract.

```python
import json
from infermesh import LMClient

with open("results.jsonl", "w") as out, \
     LMClient(model="openai/gpt-4.1-mini", max_parallel_requests=32) as client:

    def save(index: int, result, error) -> None:
        row = {"index": index}
        if error is not None:
            row["error"] = str(error)
        else:
            row["output_text"] = result.output_text
        out.write(json.dumps(row) + "\n")
        out.flush()

    client.generate_batch(prompts, on_result=save)
```

The CLI automates this with `--resume` — see the CLI section above and the
[User Guide](docs/guide.md) for the full checkpoint/resume pattern.

</details>

<details>
<summary>Rate limiting</summary>

Pass any combination of `rpm` / `tpm` / `rpd` / `tpd` to activate the built-in rate
limiter. The client queues requests automatically and respects all four limits
simultaneously.

```python
client = LMClient(
    model="openai/gpt-4.1-mini",
    rpm=500,      # requests per minute
    tpm=100_000,  # tokens per minute
)
```

Find your tier's limits in the provider dashboard: for OpenAI check **Settings →
Limits**; for Anthropic check **Console → Settings → Limits**.

Use `max_request_burst` / `max_token_burst` to allow short bursts above the steady-state
rate (token-bucket algorithm). Use `default_output_tokens` to pre-reserve output tokens
for rate-limit accounting when you don't set `max_tokens` per request.

Provider rate-limit headers (`x-ratelimit-*`) are read automatically after each response
to keep the client's internal counters in sync with the server's view. Use
`header_bucket_scope` to control whether headers are routed to the per-minute or
per-day buckets.

CLI flags: `--rpm`, `--tpm`, `--rpd`, `--tpd`, `--max-request-burst`, `--max-token-burst`.

</details>

<details>
<summary>Multi-replica routing (vLLM / SGLang)</summary>

When you run multiple inference servers for the same model, pass a `deployments` dict
to spread load across them. `model` is the logical name the router exposes; each
`DeploymentConfig.model` is the backend string sent to that server.

```python
from infermesh import DeploymentConfig, LMClient

client = LMClient(
    model="llama-3-8b",
    deployments={
        "gpu-0": DeploymentConfig(
            model="hosted_vllm/meta-llama/Meta-Llama-3-8B-Instruct",
            api_base="http://host1:8000/v1",
        ),
        "gpu-1": DeploymentConfig(
            model="hosted_vllm/meta-llama/Meta-Llama-3-8B-Instruct",
            api_base="http://host2:8000/v1",
        ),
        "gpu-2": DeploymentConfig(
            model="hosted_vllm/meta-llama/Meta-Llama-3-8B-Instruct",
            api_base="http://host3:8000/v1",
        ),
    },
    routing_strategy="least-busy",   # or "simple-shuffle" (default), "latency-based-routing"
)

result = client.generate("Summarise this paper in one paragraph.")
print(result.metrics.deployment)  # e.g. "gpu-1"
```

`DeploymentConfig` is a plain dataclass, so it maps naturally to Hydra / OmegaConf
structured config. Deployment keys (`"gpu-0"` etc.) are free-form labels.

**CLI — repeated `--api-base` flags:**

```bash
infermesh generate \
  --model llama-3-8b \
  --api-base http://host1:8000/v1 \
  --api-base http://host2:8000/v1 \
  --api-base http://host3:8000/v1 \
  --prompt "Hello"
```

**CLI — TOML file for more control:**

```toml
# deployments.toml
[deployments.gpu-0]
model = "hosted_vllm/meta-llama/Meta-Llama-3-8B-Instruct"
api_base = "http://host1:8000/v1"

[deployments.gpu-1]
model = "hosted_vllm/meta-llama/Meta-Llama-3-8B-Instruct"
api_base = "http://host2:8000/v1"
```

```bash
infermesh generate \
  --model llama-3-8b \
  --deployments-toml deployments.toml \
  --prompt "Hello"
```

Keep API keys out of TOML files — use environment variables or `--env-file` instead.

</details>

<details>
<summary>Async API</summary>

All methods have async counterparts prefixed with `a`. The sync methods work in
notebooks and scripts by running a background event loop thread — you don't need to
manage the event loop yourself.

```python
import asyncio
from infermesh import LMClient

async def main():
    async with LMClient(model="openai/gpt-4.1-mini") as client:
        result = await client.agenerate("Hello")
        batch  = await client.agenerate_batch(["prompt A", "prompt B", "prompt C"])
        emb    = await client.aembed("The quick brown fox")
        embs   = await client.aembed_batch(["text a", "text b"])
        txs    = await client.atranscribe_batch(["a.wav", "b.wav"])

asyncio.run(main())
```

`async with` calls `close()` automatically. For sync code, use `with` or call
`client.close()` when done.

</details>

<details>
<summary>Structured output</summary>

Pass a Pydantic model as `response_format` and the output is parsed automatically:

```python
from pydantic import BaseModel

class Answer(BaseModel):
    value: int
    confidence: float

result = client.generate(
    "What is 2 + 2? Respond in JSON.",
    response_format=Answer,
)

print(result.output_text)   # raw JSON string
print(result.output_parsed) # Answer(value=4, confidence=0.99)
```

A plain `dict` (JSON schema) is also accepted in place of a Pydantic model; the output
is returned as a plain Python object. Parse failures are logged as warnings;
`output_parsed` is `None` if parsing fails.

</details>

<details>
<summary>Automatic retries</summary>

By default, `LMClient` retries transient provider errors up to 3 times with
exponential backoff. This covers 429 rate-limit spikes, 503 unavailability,
500 server errors, network failures, and timeouts.

```python
client = LMClient(
    model="openai/gpt-4.1-mini",
    max_retries=3,   # default; set to 0 to disable
)
```

Backoff formula: `min(2 ** attempt, 60)` seconds plus up to 0.5 s jitter. If the
provider returns a `Retry-After` header its value is used instead (capped at 60 s).
Non-transient errors (`BadRequestError`, `AuthenticationError`, etc.) are not retried.

```python
result = client.generate("Hello")
print(result.metrics.retries)   # 0 on first-attempt success
```

CLI flag: `--max-retries`.

</details>

<details>
<summary>Timeout and per-request overrides</summary>

Set a default timeout for every request at construction time:

```python
client = LMClient(
    model="openai/gpt-4.1-mini",
    timeout=30.0,   # seconds
)
```

Any LiteLLM keyword argument passed to a `generate` / `embed` / `transcribe` call
overrides the default for that request:

```python
result = client.generate("Hello", timeout=5.0, max_tokens=64)
```

Use `default_request_kwargs` to set persistent overrides for all requests:

```python
client = LMClient(
    model="openai/gpt-4.1-mini",
    default_request_kwargs={"max_tokens": 256, "temperature": 0.7},
)
```

</details>

<details>
<summary>Benchmarking</summary>

`infermesh bench` measures client-side throughput across a concurrency sweep. It is
intentionally a **client** benchmark — it tells you the best `max_parallel_requests`
setting for your workload, not the server's maximum capacity.

```bash
infermesh bench generate \
  --model openai/gpt-4.1-mini \
  --api-base https://api.openai.com/v1 \
  --prompt "Write a haiku." \
  --warmup 5 \
  --requests 50 \
  --output-json bench.json
```

Output:

```
c=1    rps=3.14  p50=0.401s  p95=0.412s  p99=0.420s  svc_p95=0.410s  q_p95=0.001s  err=0/50  elapsed=15.9s
c=2    rps=5.81  p50=0.470s  p95=0.487s  p99=0.501s  svc_p95=0.480s  q_p95=0.002s  err=0/50  elapsed=8.6s
recommended_max_parallel_requests=8
```

`c` is the concurrency level. `svc_p95` is the P95 of net provider response time
(excluding queue wait). `q_p95` is the P95 time a request spent in the client queue.
High `q_p95` relative to `svc_p95` means the client is the bottleneck, not the server.

Use `--input-jsonl` to benchmark with a real prompt distribution. An embedding
benchmark is available as `infermesh bench embed`.

For server-centric metrics (TTFT, TPOT, ITL, request goodput), use a dedicated server
benchmark:
[vLLM](https://docs.vllm.ai/en/latest/api/vllm/benchmarks/serve/) ·
[SGLang](https://docs.sglang.ai/developer_guide/benchmark_and_profiling.html) ·
[AIPerf](https://github.com/ai-dynamo/aiperf)

</details>

## Why Not Just Use LiteLLM?

Use LiteLLM directly if provider abstraction is the only missing piece.

`infermesh` is intentionally narrower:

- LiteLLM is the provider abstraction and request layer.
- `infermesh` adds notebook-safe sync APIs and concurrent batch helpers.
- `infermesh` preserves partial failures instead of turning a long run into one
  giant exception.
- `infermesh` adds client-side throttling and replica routing for experiment
  workloads.
- `infermesh` returns typed result objects so request metadata is easier to
  inspect programmatically.

## When Not To Use It

- You only make a few single requests.
- You already have a batching and throttling layer you trust.
- You want raw provider payloads with as little abstraction as possible.

## More Detail

- [User Guide](docs/guide.md) for the complete researcher workflow, embeddings,
  transcription, multimodal inputs, rate limiting, routing, async usage, structured
  output, and benchmarking
- [API Reference](docs/api/client.md) for method signatures and parameter docs
