"""ShareGPT row formatting shared by CPU and Unsloth training entrypoints."""


def format_sharegpt_messages(conversations: list, tokenizer) -> str:
    """Turn a ShareGPT conversation list into a single training text string."""
    messages = []
    for msg in conversations:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})
    if not messages:
        return ""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def make_sharegpt_formatting_func(tokenizer):
    """Return a batched formatting callable for TRL/Unsloth trainers."""

    def formatting_func(examples):
        conversations = examples.get("conversations", [])
        if not conversations:
            return []
        if isinstance(conversations[0], dict) and "role" in conversations[0]:
            conversations = [conversations]
        texts = []
        for convo in conversations:
            if not convo:
                continue
            text = format_sharegpt_messages(convo, tokenizer)
            if text:
                texts.append(text)
        return texts

    return formatting_func
