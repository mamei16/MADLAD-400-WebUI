"""
This module provides an interface for translation using the MADLAD-400 models.
The interface allows users to enter English text, select the target language, and choose a model.
The user will receive the translated text.
"""
import sys

import gradio as gr
from tokenizers import Tokenizer
import regex

from LangMap.langid_mapping import langid_to_language
from llama_cpp_server import LlamaServer


def translate(text: str, target_language_name: str) -> str:
    """
    Translate the input text from English to another language.

    Args:
        text (str): The input text to be translated.
        target_language_name (str): The human readable target language name.
        model_name (str): The model name for translation.

    Returns:
        str: The translated text.
    """
    # Convert the selected language name back to its corresponding language code
    target_language_code = NAME_TO_CODE_MAP.get(target_language_name)

    if target_language_code is None:
        raise ValueError(f"Unsupported language: {target_language_name}")

    text_output = ""
    for paragraph in text.split("\n"):
        sentences = filter(None, PUNCT_SPLIT_PAT.split(paragraph))

        for sentence in sentences:

            sentence = sentence.strip()
            if sentence == "":
                continue

            if sentence in PUNCT_CHAR_DICT:
                text_output += sentence
                continue

            if text_output and not text_output.endswith("\n"):
                text_output += " "

            text_input = f"{target_language_code} {sentence}"
            output_gen = server.generate_with_streaming(text_input, {"add_bos_token": True})
            for decoded_text in output_gen:
                text_output += decoded_text
                yield text_output
        text_output += "\n"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:\npython app.py <path_to_gguf>")
        exit(0)

    server = LlamaServer("madlad-400-10b-mt-q6_k.gguf")
    tokenizer = Tokenizer.from_pretrained("google/madlad400-10b-mt")

    LANGUAGE_CODES = [token for token in tokenizer.get_vocab().keys() if token in langid_to_language.keys()]

    # Mapping language codes to human readable language names
    LANGUAGE_MAP = {k: v for k, v in langid_to_language.items() if k in LANGUAGE_CODES}

    # Invert the language mapping for reverse lookup (from language name to language code)
    NAME_TO_CODE_MAP = {name: code for code, name in LANGUAGE_MAP.items()}

    # Extract the language names for the dropdown in the Gradio interface
    LANGUAGE_NAMES = list(LANGUAGE_MAP.values())


    PUNCT_SPLIT_PAT = regex.compile(
        r"(?:"
        r"(\.(?=\s\p{Lu}))"   # Latin/Cyrillic/Greek/Armenian: period + space + uppercase
        r"|(۔(?=\s\p{Lu}))"   # Arabic script has no case, but if mixed with Latin, allow uppercase
        r"|(।)"               # Devanagari danda
        r"|(॥)"               # Devanagari double danda
        r"|(。)"               # Chinese/Japanese full stop
        r"|(።)"                # Ethiopic
        r"|(།)"               # Tibetan shad
        r"|(༎)"               # Tibetan double shad
        r"|(။)"               # Burmese
        r"|(។)"               # Khmer
        r"|(ໆ)"               # Lao
        r"|(᠃)"               # Mongolian
        r"|(෴)"               # Sinhala kundaliya
        # --- Colons used in natural language ---
        r"|(:(?=\s\p{L}))"    # Basic colon
        r"|(：)"              # Fullwidth colon
        r"|(﹕)"               # Small colon
        r"|(︰)"              # Vertical colon
        r")"
    )

    PUNCT_CHARS = [".", "۔",
                    "।"
                    "॥",
                    "。",
                    "።",
                    "།",
                    "༎",
                    "။",
                    "។",
                    "ໆ",
                    "᠃",
                    "෴",
                    ":",
                    "：",
                    "﹕",
                    "︰"]

    PUNCT_CHAR_DICT = dict.fromkeys(PUNCT_CHARS)
    TITLE = "MADLAD-400 Translation"
    DESCRIPTION = """
    Translation from English to (almost) 400 languages based on [research](https://arxiv.org/pdf/2309.04662)
    by Google DeepMind and Google Research.
    """

    # Gradio components
    input_text = gr.Textbox(
        label="Text",
        placeholder="Enter text here"
    )


    target_language = gr.Dropdown(
        choices=LANGUAGE_NAMES, # Use language names instead of codes
        value="English", # Default human readable language name
        label="Target language"
    )

    output_text = gr.Textbox(label="Translation")

    # Define the Gradio interface
    demo = gr.Interface(
        fn=translate,
        inputs=[input_text, target_language],
        outputs=output_text,
        title=TITLE,
        description=DESCRIPTION,
        analytics_enabled=False
    )

    # Launch the Gradio interface
    demo.launch(inbrowser=True)
