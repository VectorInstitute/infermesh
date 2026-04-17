# User Guide

## Batch Workflow

Set the relevant provider key in your environment before calling hosted APIs:

```bash
export OPENAI_API_KEY=sk-...
```

The main workflow is "run a batch, keep the good results, inspect the
failures, and retry only what failed."

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

for prompt, result, error in zip(prompts, batch.results, batch.errors or []):
    if error is not None:
        print(f"FAILED: {prompt}\n  {error}")
        retry_prompts.append(prompt)
        continue

    print(result.output_text)
    if result.token_usage is not None:
        print("tokens:", result.token_usage.total_tokens)

if retry_prompts:
    with LMClient(model="openai/gpt-4.1-mini") as retry_client:
        retry_batch = retry_client.generate_batch(retry_prompts)
```

By default, one failing request does not abort the whole batch. Failed items are
stored as `None` in `batch.results`, and the corresponding exception is stored
in `batch.errors[i]`.

For large Python batches, set `max_parallel_requests` explicitly. That enables
bounded in-flight scheduling for `generate_batch`; when it is unset, the method
may start work for the full batch up front.

### Crash-Resilient Batches with `on_result`

For large batches, you may want to write results to disk as each request
completes rather than waiting for the whole batch to finish. This way a
process crash or interruption only loses the in-flight requests, not
everything already completed.

Pass an `on_result` callback to `generate_batch` (or `agenerate_batch`):

```python
import json
from pathlib import Path
from infermesh import LMClient

prompts = [...]  # large list

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

The callback receives:

| Argument | Type | Notes |
|---|---|---|
| `index` | `int` | Position in `input_batch` |
| `result` | `GenerationResult \| None` | `None` on failure |
| `error` | `BaseException \| None` | `None` on success |

To resume from a partial output file, read the completed indices before the
batch starts and filter the input:

```python
done = set()
output_path = Path("results.jsonl")
if output_path.exists():
    for line in output_path.read_text().splitlines():
        row = json.loads(line)
        if "index" in row:
            done.add(row["index"])

pending = [(i, p) for i, p in enumerate(prompts) if i not in done]

with open(output_path, "a") as out, \
     LMClient(model="openai/gpt-4.1-mini", max_parallel_requests=32) as client:

    def save(batch_idx: int, result, error) -> None:
        orig_idx = pending[batch_idx][0]
        row = {"index": orig_idx}
        if error is not None:
            row["error"] = str(error)
        else:
            row["output_text"] = result.output_text
        out.write(json.dumps(row) + "\n")
        out.flush()

    client.generate_batch([p for _, p in pending], on_result=save)
```

The CLI `--resume` flag automates this pattern end-to-end.

Pass `return_exceptions=False` if you want the first failure to cancel the rest
of the batch and raise immediately instead.

This code works in Jupyter notebooks without any `asyncio` setup. The sync API
runs a background event loop so you do not have to.

The `model` string uses LiteLLM's `provider/model-name` format. See the
[LiteLLM model list](https://docs.litellm.ai/docs/providers) for all supported
providers and model names:

| Provider | Example |
|---|---|
| OpenAI | `"openai/gpt-4.1-mini"` |
| Anthropic | `"anthropic/claude-3-5-sonnet-20241022"` |
| Local vLLM | `"hosted_vllm/meta-llama/Meta-Llama-3-8B-Instruct"` |

`api_base` is optional for hosted providers — LiteLLM already knows their
endpoints. Set it explicitly for local servers or custom deployments.

## CLI Batch From JSONL

For dataset-driven jobs, the CLI can read one JSON object per line.

`prompts.jsonl`:

```json
{"prompt": "Summarize abstract A in one sentence."}
{"prompt": "Summarize abstract B in one sentence."}
{"prompt": "Summarize abstract C in one sentence."}
```

Run the batch:

```bash
infermesh generate \
  --model openai/gpt-4.1-mini \
  --api-base https://api.openai.com/v1 \
  --input-jsonl prompts.jsonl \
  --output-jsonl results.jsonl
```

Each output line contains a result or an error, plus an `_index` field that
records the row's position in the input file:

```json
{"_index": 0, "output_text": "Abstract A is about...", "error": null}
{"_index": 1, "output_text": null, "error": "RateLimitError(...)"}
{"_index": 2, "output_text": "Abstract C is about...", "error": null}
```

Input rows for `infermesh generate` may contain any of the following fields:

- `prompt` for a plain string prompt
- `messages` for a pre-built chat conversation
- `responses_input` for an OpenAI Responses-style input payload

### Resuming an Interrupted Run

If a long batch is interrupted (Ctrl-C, OOM, network loss), the output file
contains all rows completed so far. Re-run with `--resume` to skip them and
append only the remaining rows:

```bash
# First attempt — interrupted partway through
infermesh generate \
  --model openai/gpt-4.1-mini \
  --api-base https://api.openai.com/v1 \
  --input-jsonl prompts.jsonl \
  --output-jsonl results.jsonl

# Resume — reads results.jsonl, skips completed _index values, appends the rest
infermesh generate \
  --model openai/gpt-4.1-mini \
  --api-base https://api.openai.com/v1 \
  --input-jsonl prompts.jsonl \
  --output-jsonl results.jsonl \
  --resume
```

Results are written to disk one row at a time as each request completes, so
a crash only loses the requests that were in-flight at that moment.
`--resume` requires `--output-jsonl`.

## Generate Text

```python
result = client.generate("Say hello in one sentence.")

print(result.output_text)     # generated text
print(result.token_usage)     # prompt / completion / total token counts
print(result.finish_reason)   # "stop", "length", …
print(result.request_id)      # provider-assigned ID for debugging
```

### Structured Output

Pass a Pydantic model as `response_format` and the output is parsed
automatically:

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

A plain `dict` (JSON Schema) is also accepted in place of a Pydantic model; the
output is returned as a plain Python object and validated against the schema.
Parse failures are logged as warnings and `output_parsed` is `None` if parsing
fails.

### Timeout And Per-Request Overrides

Set a default timeout for every request at construction time:

```python
client = LMClient(
    model="openai/gpt-4.1-mini",
    api_base="https://api.openai.com/v1",
    timeout=30.0,
)
```

Any LiteLLM keyword argument passed to a call overrides the default for that
request:

```python
result = client.generate("Hello", timeout=5.0, max_tokens=64)
```

Use `default_request_kwargs` to set persistent request defaults:

```python
client = LMClient(
    model="openai/gpt-4.1-mini",
    api_base="https://api.openai.com/v1",
    default_request_kwargs={"max_tokens": 256, "temperature": 0.7},
)
```

## Create Embeddings

```python
# Single string -> EmbeddingResult
result = client.embed("The quick brown fox")
print(result.embedding)

# Multiple strings -> sent in one API call
batch = client.embed_batch(["sentence one", "sentence two", "sentence three"])
vectors = [r.embedding for r in batch if r is not None]
```

## Transcribe Audio

```python
result = client.transcribe("recording.wav")
print(result.text)
print(result.language)
print(result.duration_s)
```

Audio inputs larger than 25 MB are rejected by default. Pass
`max_transcription_bytes=None` to disable the limit, or a smaller integer to
tighten it.

## Multimodal / VLM

For URL-based images, pass the OpenAI content-block dict directly:

```python
result = client.generate([{
    "role": "user",
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
    ],
}])
```

For local files or raw bytes, use [`image_block()`][infermesh.image_block] to
handle base64 encoding:

```python
from pathlib import Path

from infermesh import LMClient, image_block

with LMClient(model="openai/gpt-4o", api_base="https://api.openai.com/v1") as client:
    result = client.generate([{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this diagram in one sentence."},
            image_block(Path("diagram.png")),
            image_block(Path("photo.jpg"), detail="high"),
            image_block(raw_bytes, mime_type="image/jpeg"),
        ],
    }])
    print(result.output_text)
```

## Handling API Keys

Never pass secrets on the command line. Instead, export provider environment
variables or use `--env-file` to load a `.env` file:

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

```bash
infermesh generate \
  --model openai/gpt-4.1-mini \
  --api-base https://api.openai.com/v1 \
  --env-file .env \
  --prompt "Hello"
```

Add `.env` to `.gitignore` so secrets are never committed.

## Rate Limiting

Pass any combination of `rpm` / `tpm` / `rpd` / `tpd` to activate the built-in
rate limiter. The client queues requests automatically and respects all four
limits simultaneously. Find your tier's limits in the provider dashboard: for
OpenAI check **Settings → Limits**; for Anthropic check **Console → Settings →
Limits**.

```python
client = LMClient(
    model="openai/gpt-4.1-mini",
    api_base="https://api.openai.com/v1",
    rpm=500,
    tpm=100_000,
)
```

Use `max_request_burst` / `max_token_burst` to allow short bursts above the
steady-state rate. Use `default_output_tokens` to pre-reserve output tokens for
rate-limit accounting when you do not set `max_tokens` per request.

Provider rate-limit headers (`x-ratelimit-*`) are read automatically after each
response to keep the client's internal counters in sync with the server's view.
Use `header_bucket_scope` to control whether headers are routed to the
per-minute or per-day buckets.

CLI flags: `--rpm`, `--tpm`, `--rpd`, `--tpd`, `--max-request-burst`,
`--max-token-burst`.

## Multi-Replica Routing

When you run multiple inference servers for the same model, pass a
`deployments` dict to spread load across them. `model` is the logical model
name the router exposes; each `DeploymentConfig.model` is the backend string
sent to that replica.

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

result = client.generate("Summarize this paper in one paragraph.")
print(result.metrics.deployment)
```

### CLI With Repeated `--api-base`

```bash
infermesh generate \
  --model llama-3-8b \
  --api-base http://host1:8000/v1 \
  --api-base http://host2:8000/v1 \
  --api-base http://host3:8000/v1 \
  --prompt "Hello"
```

### CLI With TOML

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

Keep API keys out of TOML files. Use environment variables or `--env-file`
instead.

## Automatic Retries

By default, `LMClient` retries transient provider errors up to **3 times** with
exponential backoff. This covers the failure modes you'd otherwise have to handle
yourself: rate-limit spikes, momentary server unavailability, network drops, and
request timeouts.

```python
client = LMClient(
    model="openai/gpt-4.1-mini",
    max_retries=3,   # default; set to 0 to disable
)
```

**What gets retried:**

| Status | Exception |
|---|---|
| 429 Too Many Requests | `RateLimitError` |
| 503 Service Unavailable | `ServiceUnavailableError` |
| 500 Internal Server Error | `InternalServerError` |
| Network failure | `APIConnectionError` |
| Request timeout | `Timeout` |

**What is not retried** (propagates immediately): `BadRequestError`,
`AuthenticationError`, `PermissionDeniedError`, `NotFoundError`,
`ContextWindowExceededError`, and other non-transient errors.

**Backoff formula:** `min(2 ** attempt, 60)` seconds plus up to 0.5 s of random
jitter. If the provider returns a `Retry-After` header the client sleeps for that
duration instead (capped at 60 s).

The backoff sleep happens **outside** the semaphore, so other in-flight requests
are not blocked while one request is waiting to retry.

Use `result.metrics.retries` to see how many attempts were needed:

```python
result = client.generate("Hello")
if result.metrics.retries > 0:
    print(f"Succeeded after {result.metrics.retries} retries")
```

For batch jobs, retries are per-item and transparent — the result you get back
already reflects the final successful response.

CLI flag: `--max-retries`.

## Async API

All methods have async counterparts prefixed with `a`. The sync methods work in
notebooks and scripts by running a background event loop thread, so you do not
need to manage the loop yourself.

```python
import asyncio
from infermesh import LMClient

async def main():
    async with LMClient(model="openai/gpt-4.1-mini", api_base="https://api.openai.com/v1") as client:
        result = await client.agenerate("Hello")
        batch = await client.agenerate_batch(["prompt A", "prompt B", "prompt C"])
        embedding = await client.aembed("The quick brown fox")
        embedding_batch = await client.aembed_batch(["text a", "text b"])
        print(result.output_text, len(batch), len(embedding.embedding), len(embedding_batch))

asyncio.run(main())
```

`async with` calls `close()` automatically. For sync code, use `with` or call
`client.close()` when done.

## Benchmarking

`infermesh bench` measures client-side throughput across a sweep. It is
intentionally a client benchmark: it helps you choose a good
`max_parallel_requests` or embedding batch size for your workload, not the
server's absolute maximum capacity.

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

`c` is the concurrency level. `p50`/`p95`/`p99` are end-to-end latency
percentiles as seen by the caller. `svc_p95` is the P95 of net provider
response time (excluding queue wait). `q_p95` is the P95 of time a request
spent waiting in the client queue before being sent. High `q_p95` relative to
`svc_p95` means the client is the bottleneck, not the server.

Use `--input-jsonl` to benchmark with a real prompt distribution instead of one
repeated prompt. An embedding benchmark is available as `infermesh bench embed`.
