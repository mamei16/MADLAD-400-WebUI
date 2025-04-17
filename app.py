import os
import tempfile
import shutil
import torch
import gradio as gr
from pathlib import Path

# Docling imports
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption, SimplePipeline

# LangChain imports for document splitting
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Transformers imports for translation
import spaces
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from LangMap.langid_mapping import langid_to_language

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "google/madlad400-3b-mt"

# Load model and tokenizer once at the beginning
print(f"Loading MADLAD-400 3B model on {DEVICE}...")
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
MODEL = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32,
    low_cpu_mem_usage=True
)
MODEL.to(DEVICE)
print("Model loaded successfully")

# Get language codes and names
LANGUAGE_CODES = [token for token in TOKENIZER.get_vocab().keys() if token in langid_to_language.keys()]
LANGUAGE_MAP = {k: v for k, v in langid_to_language.items() if k in LANGUAGE_CODES}
NAME_TO_CODE_MAP = {name: code for code, name in LANGUAGE_MAP.items()}
LANGUAGE_NAMES = list(LANGUAGE_MAP.values())

# Function to determine document format
def get_document_format(file_path) -> InputFormat:
    """Determine the document format based on file extension"""
    try:
        file_path = str(file_path)
        extension = os.path.splitext(file_path)[1].lower()

        format_map = {
            '.pdf': InputFormat.PDF,
            '.docx': InputFormat.DOCX,
            '.doc': InputFormat.DOCX,
            '.pptx': InputFormat.PPTX,
            '.html': InputFormat.HTML,
            '.htm': InputFormat.HTML
        }
        return format_map.get(extension, None)
    except Exception as e:
        return f"Error in get_document_format: {str(e)}"

# Function to convert document to markdown
def convert_document_to_markdown(doc_path) -> str:
    """Convert document to markdown using simplified pipeline"""
    try:
        # Convert to absolute path string
        input_path = os.path.abspath(str(doc_path))
        print(f"Converting document: {doc_path}")
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy input file to temp directory
            temp_input = os.path.join(temp_dir, os.path.basename(input_path))
            shutil.copy2(input_path, temp_input)
            
            # Configure pipeline options
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False  # Disable OCR temporarily
            pipeline_options.do_table_structure = True
            
            # Create converter with minimal options
            converter = DocumentConverter(
                allowed_formats=[
                    InputFormat.PDF,
                    InputFormat.DOCX,
                    InputFormat.HTML,
                    InputFormat.PPTX,
                ],
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                    ),
                    InputFormat.DOCX: WordFormatOption(
                        pipeline_cls=SimplePipeline
                    )
                }
            )
            
            # Convert document
            print("Starting conversion...")
            conv_result = converter.convert(temp_input)
            
            if not conv_result or not conv_result.document:
                raise ValueError(f"Failed to convert document: {doc_path}")
            
            # Export to markdown
            print("Exporting to markdown...")
            md = conv_result.document.export_to_markdown()
            
            # Create output path
            output_dir = os.path.dirname(input_path)
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            md_path = os.path.join(output_dir, f"{base_name}_converted.md")
            
            # Write markdown file
            print(f"Writing markdown to: {base_name}_converted.md")
            with open(md_path, "w", encoding="utf-8") as fp:
                fp.write(md)
            
            return md_path, md
    except Exception as e:
        return None, f"Error converting document: {str(e)}"

# Function to split markdown into chunks
def split_markdown_document(markdown_text, chunk_size=2000, chunk_overlap=200):
    """Split markdown document into manageable chunks for translation"""
    # Define headers to split on
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    
    # First try splitting by headers
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    header_splits = markdown_splitter.split_text(markdown_text)
    
    # Then split by character if needed to ensure chunk size
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    if header_splits:
        # Further split any sections that are too large
        final_chunks = []
        for doc in header_splits:
            # Check if the chunk is larger than our desired size
            if len(doc.page_content) > chunk_size:
                smaller_chunks = text_splitter.split_text(doc.page_content)
                # Add header metadata to each smaller chunk
                for chunk in smaller_chunks:
                    chunk_with_metadata = {
                        "content": chunk,
                        "metadata": doc.metadata
                    }
                    final_chunks.append(chunk_with_metadata)
            else:
                chunk_with_metadata = {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                final_chunks.append(chunk_with_metadata)
    else:
        # If no headers, just split by character
        text_chunks = text_splitter.split_text(markdown_text)
        final_chunks = [{"content": chunk, "metadata": {}} for chunk in text_chunks]
    
    return final_chunks

# Translation function using the pre-loaded model
def translate_text(text: str, target_language_name: str) -> str:
    """Translate the input text from English to another language using the pre-loaded model."""
    try:
        # Convert the selected language name back to its corresponding language code
        target_language_code = NAME_TO_CODE_MAP.get(target_language_name)
        if target_language_code is None:
            raise ValueError(f"Unsupported language: {target_language_name}")
        
        # Prepare input for the model
        text = target_language_code + text
        
        # Handle potential CUDA out of memory issues
        try:
            input_ids = TOKENIZER(text, return_tensors="pt").input_ids.to(DEVICE)
            
            # Generate translation with reduced memory footprint
            with torch.no_grad():  # Disable gradient calculation to save memory
                outputs = MODEL.generate(
                    input_ids=input_ids,
                    max_new_tokens=1024,  # Limiting tokens to avoid memory issues
                    do_sample=False,
                    num_beams=2  # Use fewer beams to reduce memory usage
                )
            
            # Decode the output
            text_translated = TOKENIZER.batch_decode(outputs, skip_special_tokens=True)
            return text_translated[0]
            
        except torch.cuda.OutOfMemoryError:
            # Fall back to CPU if CUDA runs out of memory
            print("CUDA out of memory, falling back to CPU for this chunk")
            # Move tensors to CPU
            if DEVICE.type == 'cuda':
                input_ids = TOKENIZER(text, return_tensors="pt").input_ids
                model_cpu = MODEL.to('cpu')
                with torch.no_grad():
                    outputs = model_cpu.generate(
                        input_ids=input_ids,
                        max_new_tokens=1024,
                        do_sample=False,
                        num_beams=1
                    )
                # Move model back to GPU
                MODEL.to(DEVICE)
                text_translated = TOKENIZER.batch_decode(outputs, skip_special_tokens=True)
                return text_translated[0]
            else:
                raise  # Re-raise if not on CUDA
    
    except Exception as e:
        print(f"Translation error: {str(e)}")
        # Return error message as translation result
        return f"[Translation Error: {str(e)}]"

def translate_chunks(chunks, target_language_name, progress=None):
    """Translate all chunks and maintain their structure"""
    translated_chunks = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        try:
            content = chunk["content"]
            metadata = chunk["metadata"]
            
            # Update progress before translation (to show we're working on this chunk)
            if progress is not None:
                progress((i) / total_chunks, f"Translating chunk {i+1}/{total_chunks}")
            
            # Translate the content - limit chunk size if it's very large
            if len(content) > 4000:
                content = content[:4000]  # Limit very large chunks to avoid memory issues
                
            translated_content = translate_text(content, target_language_name)
            
            # Store with original metadata
            translated_chunks.append({
                "content": translated_content,
                "metadata": metadata
            })
            
            # Update progress after translation is complete
            if progress is not None:
                progress((i + 1) / total_chunks, f"Translated chunk {i+1}/{total_chunks}")
                
        except Exception as e:
            import traceback
            error_message = f"Error translating chunk {i+1}: {str(e)}\n{traceback.format_exc()}"
            print(error_message)
            
            # Add error message as content
            translated_chunks.append({
                "content": f"[Translation Error in Chunk {i+1}: {str(e)}]",
                "metadata": metadata if 'metadata' in locals() else {}
            })
            
            # Update progress to show error but still continue
            if progress is not None:
                progress((i + 1) / total_chunks, f"Error in chunk {i+1}/{total_chunks} - continuing...")
    
    return translated_chunks

def reconstruct_markdown(translated_chunks):
    """Reconstruct the translated chunks into a single markdown document"""
    result = []
    
    for chunk in translated_chunks:
        content = chunk["content"]
        metadata = chunk["metadata"]
        
        # Add headers if they exist in metadata
        if "Header 1" in metadata:
            result.append(f"# {metadata['Header 1']}")
        if "Header 2" in metadata:
            result.append(f"## {metadata['Header 2']}")
        if "Header 3" in metadata:
            result.append(f"### {metadata['Header 3']}")
        if "Header 4" in metadata:
            result.append(f"#### {metadata['Header 4']}")
        
        # Add the translated content
        result.append(content)
    
    return "\n\n".join(result)

# Main processing function for Gradio
@spaces.GPU
def process_document_for_translation(file_obj, target_language_name, chunk_size, chunk_overlap, progress=gr.Progress()):
    """Main function to process document for translation"""
    try:
        print(f"Starting document translation to {target_language_name}")
        print(f"Chunk size: {chunk_size}, Chunk overlap: {chunk_overlap}")
        
        # Handle file object based on type
        if isinstance(file_obj, str):
            # If it's a string path
            temp_path = file_obj
        else:
            # Create temp file and save uploaded content
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(os.path.basename(file_obj.name))[1]) as temp_file:
                temp_path = temp_file.name
                # Save the uploaded file content
                with open(file_obj.name, 'rb') as f:
                    shutil.copyfileobj(f, temp_file)
        
        progress(0.1, "Document uploaded")
        
        # Convert document to markdown
        md_path, md_content = convert_document_to_markdown(temp_path)
        if md_path is None:
            return None, md_content  # Return error message
        
        progress(0.3, "Document converted to markdown")
        
        # Split markdown into chunks
        chunks = split_markdown_document(md_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"Document split into {len(chunks)} chunks")
        
        progress(0.4, "Document split into chunks")
        
        # Translate chunks
        translated_chunks = translate_chunks(chunks, target_language_name, progress)
        
        progress(0.9, "Translation completed")
        
        # Reconstruct markdown
        translated_markdown = reconstruct_markdown(translated_chunks)
        
        # Save translated markdown to file
        base_name = os.path.splitext(os.path.basename(temp_path))[0]
        translated_file_path = os.path.join(tempfile.gettempdir(), f"{base_name}_translated_{target_language_name}.md")
        
        with open(translated_file_path, "w", encoding="utf-8") as f:
            f.write(translated_markdown)
        
        progress(1.0, "Translation saved")
        
        # Clean up if we created a temp file
        if temp_path != file_obj and os.path.exists(temp_path):
            os.unlink(temp_path)
        
        return translated_file_path, "Translation completed successfully!"
        
    except Exception as e:
        import traceback
        error_message = f"Error processing document: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        return None, error_message

# Create Gradio interface
def create_app():
    with gr.Blocks(title="Document Translation App") as app:
        gr.Markdown("# Document Translation with MADLAD-400")
        gr.Markdown("""
        This application translates documents (PDF, DOCX, PPTX, HTML) from English to almost 400 languages
        using Google's MADLAD-400 3B translation model.
        
        1. Upload your document
        2. Select the target language
        3. Configure chunking parameters
        4. Click 'Translate' to get your translated document
        """)
        
        with gr.Row():
            with gr.Column():
                file_input = gr.File(label="Upload Document (PDF, DOCX, PPTX, HTML)")
                
                target_language = gr.Dropdown(
                    choices=LANGUAGE_NAMES,
                    value="French",
                    label="Target Language"
                )
                
                with gr.Row():
                    chunk_size = gr.Slider(
                        minimum=500,
                        maximum=4000,
                        value=2000,
                        step=100,
                        label="Chunk Size (characters)"
                    )
                    
                    chunk_overlap = gr.Slider(
                        minimum=50,
                        maximum=500,
                        value=200,
                        step=50,
                        label="Chunk Overlap (characters)"
                    )
                
                translate_btn = gr.Button("Translate Document", variant="primary")
            
            with gr.Column():
                output_message = gr.Textbox(label="Status")
                output_file = gr.File(label="Translated Document")
        
        # Connect the components
        translate_btn.click(
            fn=process_document_for_translation,
            inputs=[file_input, target_language, chunk_size, chunk_overlap],
            outputs=[output_file, output_message]
        )
    
    return app

# Create and launch the application
if __name__ == "__main__":
    app = create_app()
    app.launch()
