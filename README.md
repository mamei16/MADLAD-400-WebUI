<img width="2060" height="522" alt="Image" src="https://github.com/user-attachments/assets/650baa55-341c-4717-9bb2-508716c2543a" />

### Features
This app automatically splits longer texts into paragraphs and sentences before translating, thus overcoming the short maximum text length of the MADLAD-400 models.



### How to run
1. Install https://github.com/mamei16/llama-cpp-binaries (This installs [a fork](https://github.com/fairydreaming/llama.cpp/tree/t5-server) of llama.cpp, which adds support for madlad models to the llama-server.)
1. Install the other requirements
2. Download a GGUF version of a madlad400 model, for example from the official repo: https://huggingface.co/google/madlad400-10b-mt/blob/main/model-q6k.gguf
4. Launch the app with: `python app.py <path_to_your_gguf>`


### VRAM Requirements
If you run an 8-bit quantized version of the 10B model (such as the one linked above), it can be run with 12GB of VRAM.