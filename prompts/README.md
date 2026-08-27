# GPT zero-shot prompts

One file per task. Each file has two placeholders the benchmark script fills in:

- `{{TEXT}}` — the full sample text (chat-templated conversation, possibly with `<think>` block, etc.)

Expected model output: the first token should be either `yes` or `no` (case-insensitive). The script maps:
- `yes` → label=1 (the "positive" / problem class: incorrect / different topic / conflict / repetitive / branched / unsafe)
- `no`  → label=0 (the "negative" / clean class)

Label conventions per task (matches training caches):

| Task | label=0 | label=1 |
|---|---|---|
| adversarial_if | correct | incorrect |
| wildchat | same_topic | different_topic |
| verih | aligned | conflict |
| cot_repetitive | not_repetitive | repetitive |
| branched_thinking | not_branched | branched |
| nemotron_safety | safe | unsafe |
