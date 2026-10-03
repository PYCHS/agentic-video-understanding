# Qwen3-VL backend (not yet tested with weights)

qwen.py implements the Backend interface for Qwen/Qwen3-VL-8B-Instruct. It follows the
[official model-card flow](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct): apply the chat
template, generate, remove the input prefix, then decode the generated tokens.

QwenSettings requires full model and processor commit hashes, minimum/maximum processor
pixels and an output-token limit. These values are not filled into the evaluation config
yet. The initial implementation uses one explicit CUDA device, BF16 and no quantization.
It requests greedy decoding with one beam. The pilot backbone needs a separate adapter.

QwenBackend.load defaults to local_files_only=True. It does not download weights unless the
caller explicitly passes False. torch and transformers are imported only during loading;
CPU foundation tests do not require either. See requirements/inference.txt for the still
unlocked dependency skeleton. Validate compatible versions before creating an environment lock.

Per-call latency includes chat templating, transfer, generation and text decoding, bracketed
by CUDA synchronization. It excludes model loading, video reading and prompt construction.
Input tokens include the entire processed input. Output tokens count generated IDs, including
terminal special tokens. Visual tokens count occurrences of the model's image_token_id in
the processed input; if that ID is unavailable, the count is null. This definition counts
repeated image context on every call and needs checking against a real processor tensor.

Current tests use interface doubles. They check greedy settings, prompt-prefix removal,
token accounting and error propagation. They do not establish model loading, actual image
handling, GPU fit or prediction quality. There are no automatic retries or timeouts here;
the existing call wrapper records a backend exception and ends the question.

Local readiness check on 2026-10-03 found an RTX 5090 Laptop GPU reporting 24463 MiB, but
the project environment had neither torch nor transformers installed. No weights were loaded.
Next: create a compatible inference environment, pin revisions, check a small image prompt,
then test 8/16/24-frame memory use before scoring. Do not assume the GPU fits every setting.
