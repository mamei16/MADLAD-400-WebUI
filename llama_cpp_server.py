import json
import os
import pprint
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, List
from logging import Logger

import llama_cpp_binaries
import requests

from utils import resolve_model_path

llamacpp_valid_cache_types = {"fp16", "q8_0", "q4_0"}

logger = Logger(__name__)

class LlamaServer:
    def __init__(self, model_path, server_path=None):
        """
        Initialize and start a server for llama.cpp models.
        """
        self.model_path = model_path
        self.server_path = server_path
        self.port = self._find_available_port()
        self.process = None
        self.session = requests.Session()
        self.vocabulary_size = None
        self.bos_token = "<s>"
        self.last_prompt_token_count = 0

        # Start the server
        self._start_server()

    def encode(self, text, add_bos_token=False, **kwargs):
        if self.bos_token and text.startswith(self.bos_token):
            add_bos_token = False

        url = f"http://127.0.0.1:{self.port}/tokenize"
        payload = {
            "content": text,
            "add_special": add_bos_token,
        }

        response = self.session.post(url, json=payload)
        result = response.json()
        return result.get("tokens", [])

    def decode(self, token_ids, **kwargs):
        url = f"http://127.0.0.1:{self.port}/detokenize"
        payload = {
            "tokens": token_ids,
        }

        response = self.session.post(url, json=payload)
        result = response.json()
        return result.get("content", "")

    def prepare_payload(self, state):
        payload = {
            "temperature": 0,
            "dynatemp_range": 0,
            "top_k": 1,
            "top_p": 0.,
            "min_p": 0.0,
            "top_n_sigma": -1,
            "typical_p": 1.0,
            "repeat_penalty": 1,
            "repeat_last_n": 64,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "dry_multiplier": 0.0,
            "dry_base": 1.75,
            "dry_allowed_length": 2,
            "dry_penalty_last_n": 4096,
            "xtc_probability": 0.0,
            "xtc_threshold": 0.1,
            "mirostat": 0,
            "mirostat_tau": 0.1,
            "mirostat_eta": 5.0,
        }

        return payload


    def generate_with_streaming(self, prompt, state=None):
        url = f"http://127.0.0.1:{self.port}/completion"
        payload = self.prepare_payload(state)


        # Text only case
        token_ids = self.encode(prompt, add_bos_token=state["add_bos_token"])
        payload["prompt"] = token_ids

        payload.update({
            #"n_predict": max_new_tokens,
            "stream": True,
            "cache_prompt": False
        })

        # Make the generation request
        response = self.session.post(url, json=payload, stream=True)
        try:
            if response.status_code == 400 and response.json()["error"]["type"] == "exceed_context_size_error":
                logger.error("The request exceeds the available context size, try increasing it")
            else:
                response.raise_for_status()  # Raise an exception for HTTP errors

            is_first_line = True
            # Process the streaming response
            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    line = line.decode('utf-8')

                    # Check if the line starts with "data: " and remove it
                    if line.startswith('data: '):
                        line = line[6:]  # Remove the "data: " prefix

                    # Parse the JSON data
                    data = json.loads(line)

                    # Extract the token content
                    if data.get('content', ''):
                        if is_first_line:
                            yield data['content'].lstrip()
                            is_first_line = False
                        else:
                            yield data['content']

                    # Check if generation is complete
                    if data.get('stop', False):
                        break

                except json.JSONDecodeError as e:
                    # Log the error and the problematic line
                    print(f"JSON decode error: {e}")
                    print(f"Problematic line: {line}")
                    continue
        finally:
            response.close()

    def generate(self, prompt, state):
        output = ""
        for output in self.generate_with_streaming(prompt, state):
            pass

        return output

    def _get_vocabulary_size(self):
        """Get and store the model's maximum context length."""
        url = f"http://127.0.0.1:{self.port}/v1/models"
        response = self.session.get(url).json()

        if "data" in response and len(response["data"]) > 0:
            model_info = response["data"][0]
            if "meta" in model_info and "n_vocab" in model_info["meta"]:
                self.vocabulary_size = model_info["meta"]["n_vocab"]

    def _get_bos_token(self):
        """Get and store the model's BOS token."""
        url = f"http://127.0.0.1:{self.port}/props"
        response = self.session.get(url).json()
        if "bos_token" in response:
            self.bos_token = response["bos_token"]

    def _find_available_port(self):
        """Find an available port by letting the OS assign one."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))  # Bind to port 0 to get an available port
            return s.getsockname()[1]

    def _start_server(self):
        """Start the llama.cpp server and wait until it's ready."""
        # Determine the server path
        if self.server_path is None:
            self.server_path = llama_cpp_binaries.get_binary_path()

        # Build the command
        cmd = [
            self.server_path,
            "--model", self.model_path,
            #"--ctx-size", str(shared.args.ctx_size),
            "--gpu-layers", "999",
            #"--batch-size", str(shared.args.batch_size),
            "--port", str(self.port),
            #"--no-webui",
            #"--chat-template", "template.jinja",
            #"--jinja",
        ]

        env = os.environ.copy()
        if os.name == 'posix':
            current_path = env.get('LD_LIBRARY_PATH', '')
            if current_path:
                env['LD_LIBRARY_PATH'] = f"{current_path}:{os.path.dirname(self.server_path)}"
            else:
                env['LD_LIBRARY_PATH'] = os.path.dirname(self.server_path)

        # Start the server with pipes for output
        self.process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=env
        )

        threading.Thread(target=filter_stderr_with_progress, args=(self.process.stderr,), daemon=True).start()

        # Wait for server to be healthy
        health_url = f"http://127.0.0.1:{self.port}/health"
        while True:
            # Check if process is still alive
            if self.process.poll() is not None:
                # Process has terminated
                exit_code = self.process.poll()
                raise RuntimeError(f"Server process terminated unexpectedly with exit code: {exit_code}")

            try:
                response = self.session.get(health_url)
                if response.status_code == 200:
                    break
            except:
                pass

            time.sleep(1)

        # Server is now healthy, get model info
        self._get_vocabulary_size()
        self._get_bos_token()
        return self.port

    def __enter__(self):
        """Support for context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support for context manager."""
        self.stop()

    def __del__(self):
        """Cleanup when the object is deleted."""
        self.stop()

    def stop(self):
        """Stop the server process."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

            self.process = None


def filter_stderr_with_progress(process_stderr):
    """
    Reads stderr lines from a process, filters out noise, and displays progress updates
    inline (overwriting the same line) until completion.
    """
    progress_re = re.compile(r'slot update_slots: id.*progress = (\d+\.\d+)')
    last_was_progress = False

    try:
        # Read in binary mode and decode manually
        buffer = b""
        while True:
            # Read chunks aggressively to prevent buffer overflow
            chunk = process_stderr.read(4096)
            if not chunk:
                break

            buffer += chunk

            # Process complete lines
            while b'\n' in buffer:
                line_bytes, buffer = buffer.split(b'\n', 1)
                try:
                    line = line_bytes.decode('utf-8', errors='replace').strip('\r\n')
                    if line:  # Process non-empty lines
                        match = progress_re.search(line)

                        if match:
                            progress = float(match.group(1))

                            # Extract just the part from "prompt processing" onwards
                            prompt_processing_idx = line.find('prompt processing')
                            if prompt_processing_idx != -1:
                                display_line = line[prompt_processing_idx:]
                            else:
                                display_line = line  # fallback to full line

                            # choose carriage return for in-progress or newline at completion
                            end_char = '\r' if progress < 1.0 else '\n'
                            print(display_line, end=end_char, file=sys.stderr, flush=True)
                            last_was_progress = (progress < 1.0)

                        # skip noise lines
                        elif not (line.startswith(('srv ', 'slot ')) or 'log_server_r: request: GET /health' in line):
                            # if we were in progress, finish that line first
                            if last_was_progress:
                                print(file=sys.stderr)

                            print(line, file=sys.stderr, flush=True)
                            last_was_progress = False

                except Exception:
                    continue

    except (ValueError, IOError):
        pass
    finally:
        try:
            process_stderr.close()
        except:
            pass
