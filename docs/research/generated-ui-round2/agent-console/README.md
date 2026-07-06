# Agent Console Round-2 Research

## Commercial leader

ChatGPT is the leader for general agent conversation ergonomics, especially with Projects for long-running workspaces. Claude is the strongest adjacent reference for artifact-oriented collaboration, and LangSmith is the benchmark for trace inspection and run observability.

## Leader weaknesses

- ChatGPT Projects organize long-running work, but tool-call inspection and run comparison are not exposed as generated app primitives.
- Claude Artifacts make outputs tangible, but the tool/runtime inspection path is still separate from generated business app consoles.
- LangSmith is excellent for observability, but it is a developer platform rather than an embedded business-user console.
- OpenAI streaming improves latency, but most generated internal apps do not expose stream health, prompt versions, branches, and compare views together.

## Differentiators proposed

1. Streaming Meter: show approximate tokens, character count, rate, and cost mode alongside the conversation.
2. Conversation Branching: save a local fork seed from the current message for quick alternate runs.
3. Tool-call Inspector: expose declared tools and availability without forcing users into raw JSON.
4. Prompt Library: offer versioned role/capability prompts that can be loaded into the message box.
5. Run Compare: compare prompt, result, payload, and tool/capability counts in the same console.

## Shipped verdict

APG now turns the generated agent console into an operational cockpit. Before, it had a clean chat form, stream checkbox, raw JSON, and team lanes. After, it adds meter, fork, tool inspector, prompt library, and compare cards without new dependencies or runtime URLs.
