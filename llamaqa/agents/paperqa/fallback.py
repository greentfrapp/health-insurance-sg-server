import json

FALLBACK_RESPONSE_CONTENT = "Sorry, something seems to have gone wrong."
FALLBACK_HISTORY = [
    {
        "role": "assistant",
        "formattedContent": {
            "question": "",
            "text": FALLBACK_RESPONSE_CONTENT,
            "references": [],
        },
    },
]
FALLBACK_FINAL_RESPONSE = "Final Response: " + json.dumps(FALLBACK_HISTORY)
