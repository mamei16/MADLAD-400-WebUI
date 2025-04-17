"""
This module provides an interface for translation using the MADLAD-400 models.
The interface allows users to enter English text, select the target language, and choose a model.
The user will receive the translated text.
"""

import gradio as gr
import spaces
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from LangMap.langid_mapping import langid_to_language

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the tokenizer
TOKENIZER_3B_MT = AutoTokenizer.from_pretrained("google/madlad400-3b-mt", use_fast=True)

# Retrieve the language codes
LANGUAGE_CODES = [token for token in TOKENIZER_3B_MT.get_vocab().keys() if token in langid_to_language.keys()]

# Mapping language codes to human readable language names
LANGUAGE_MAP = {k: v for k, v in langid_to_language.items() if k in LANGUAGE_CODES}

# Invert the language mapping for reverse lookup (from language name to language code)
NAME_TO_CODE_MAP = {name: code for code, name in LANGUAGE_MAP.items()}

# Extract the language names for the dropdown in the Gradio interface
LANGUAGE_NAMES = list(LANGUAGE_MAP.values())

# Model choices
MODEL_CHOICES = [
    "google/madlad400-3b-mt", 
    "google/madlad400-7b-mt", 
    "google/madlad400-10b-mt", 
    "google/madlad400-7b-mt-bt"
]

MODEL_RESOURCES = {}

def load_tokenizer_model(model_name: str):
    """
    Load tokenizer and model for a chosen model name.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        tuple: The tokenizer and model for the specified model.
    """
    if model_name not in MODEL_RESOURCES:
        # Load tokenizer and model for the first time
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16)
        model.to(DEVICE)
        MODEL_RESOURCES[model_name] = (tokenizer, model)
    return MODEL_RESOURCES[model_name]

@spaces.GPU
def translate(text: str, target_language_name: str, model_name: str) -> str:
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
    
    # Load tokenizer and model if not already loaded
    tokenizer, model = load_tokenizer_model(model_name)
    
    text = target_language_code + text
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    
    outputs = model.generate(input_ids=input_ids, max_new_tokens=128000)
    text_translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return text_translated[0]

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
    value="Hawaiian", # Default human readable language name
    label="Target language"
)

model_choice = gr.Dropdown(
    choices=MODEL_CHOICES, 
    value="google/madlad400-3b-mt", 
    label="Model"
)

output_text = gr.Textbox(label="Translation")

# Define the Gradio interface
demo = gr.Interface(
    fn=translate,
    inputs=[input_text, target_language, model_choice],
    outputs=output_text,
    title=TITLE,
    description=DESCRIPTION
)

# Launch the Gradio interface
demo.launch()
