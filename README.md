<img width="2060" height="807" alt="Image" src="https://github.com/user-attachments/assets/8051f60e-47f9-40f5-964b-1a59a6f70f2d" />

### How to run
1. Install the requirements
2. Download a converted Ctranslate2 version of a madlad400 model and all related files, for example from: https://huggingface.co/SoybeanMilk/madlad400-10b-mt-ct2-int8_float16/tree/main
3. Place all files in the same folder
4. Change the `MODEL_DIR` variable in app.py to the path of your new folder


I'm not sure if all MADLAD models share the same tokenizer, so if you use the 7B or 3B version, you may also want to change the tokenizer initialization. For the 3B version, e.g., you would change it like so:
```diff
-TOKENIZER_10B_MT = AutoTokenizer.from_pretrained("google/madlad400-10b-mt", use_fast=True)
+TOKENIZER_10B_MT = AutoTokenizer.from_pretrained("google/madlad400-3b-mt", use_fast=True)
```

### VRAM Requirements
If you run an 8-bit quantized version of the 10B model (such as the one linked above), it can be run with 12GB of VRAM.