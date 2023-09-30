import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

kernel = sk.Kernel()

api_key, org_id = sk.openai_settings_from_dot_env()
kernel.add_chat_service(
    "chat-gpt", OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id)
)

summarize = kernel.create_semantic_function(
    "{{$input}}\n\nOne line TLDR with the fewest words.",
    max_tokens=50,
    temperature=0.2,
    top_p=0.5,
)

print(
    summarize(
        input(
            "Type of paste text to summarize here and press Enter (no newline characters!): "
        )
    )
)
