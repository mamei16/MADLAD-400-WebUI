"""
This module provides an interface for translation using the MADLAD-400 models.
The interface allows users to enter English text, select the target language, and choose a model.
The user will receive the translated text.
"""

import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import ctranslate2
import regex

from LangMap.langid_mapping import langid_to_language



MODEL_DIR = "madlad400-10b-mt-ct2-int8_float16/"

# Initialize the tokenizer
TOKENIZER_10B_MT = AutoTokenizer.from_pretrained("google/madlad400-10b-mt", use_fast=True)

# Retrieve the language codes
LANGUAGE_CODES = [token for token in TOKENIZER_10B_MT.get_vocab().keys() if token in langid_to_language.keys()]

# Mapping language codes to human readable language names
LANGUAGE_MAP = {k: v for k, v in langid_to_language.items() if k in LANGUAGE_CODES}

# Invert the language mapping for reverse lookup (from language name to language code)
NAME_TO_CODE_MAP = {name: code for code, name in LANGUAGE_MAP.items()}

# Extract the language names for the dropdown in the Gradio interface
LANGUAGE_NAMES = list(LANGUAGE_MAP.values())


translator = ctranslate2.Translator(MODEL_DIR, "cuda")


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
        for sentence in regex.split("\.(?= [A-Z]|\n)", paragraph):

            sentence = sentence.strip()
            if sentence == "":
                continue

            text_input = f"{target_language_code} {sentence}"
            input_tokens = TOKENIZER_10B_MT.tokenize(text_input)

            results = translator.translate_batch([input_tokens], beam_size=1,
                                                return_scores=False,
                                                batch_type="tokens")
            output_tokens = results[0].hypotheses[0]
            decoded_text = TOKENIZER_10B_MT.decode(TOKENIZER_10B_MT.convert_tokens_to_ids(output_tokens))
            if text_output.endswith("\n"):
                text_output += decoded_text
            else:
                if not text_output.endswith('.') :
                    text_output += "."
                text_output += " " + decoded_text
            yield text_output.lstrip(" .")
        text_output += "\n"

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
