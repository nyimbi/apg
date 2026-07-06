# Raw Thinking

The existing agent console already solved the basics: prompt input, structured payload, streaming checkbox, sanitized response, and team lanes. The missing layer was observability and iteration support. The best commercial tools split this across multiple products: ChatGPT for conversation/project organization, Claude for artifacts, LangSmith for traces, and OpenAI APIs for streaming/tooling.

APG can combine a practical subset because the generated console knows the declared tools, capabilities, runtime, model, latest prompt, and latest result. The shipped implementation should keep everything static and local: no external trace service, no new backend storage, no large JavaScript bundle.

Rejected ideas:

- Persistent agent run database. Useful later, but this workspace can compare current request/result metadata without adding storage.
- Full trace tree. The generated result shape is not guaranteed to contain nested traces, so a declared-tool inspector is more honest.
- Real cost pricing. Pricing varies by model/provider and changes over time, so the UI labels the metric as an offline estimate.
