# Open WebUI empty-tools compatibility patch

The v0.11.1 OpenAI router can forward `tools: []` after Anthropic conversion or
fallback. vLLM 0.28.0 rejects this with HTTP 400. Apply
`openwebui-empty-tools.patch` to the pinned Open WebUI router and mount the
patched file read-only at `/app/backend/open_webui/routers/openai.py`.
The patch removes only an empty tool list and its associated choice flags;
nonempty tools remain unchanged. Recheck upstream compatibility on upgrades.

The A100 image profile also disables the multimodal processor cache after
repeated-image turns exposed P0/P1 cache drift. Text prefix caching remains
enabled. Repeated streaming images, not just first-image nonstream requests,
must pass before this cache is re-enabled.
