# Speech-to-Speech Project: Code Execution Flow Analysis

## Introduction

This document provides a detailed, step-by-step analysis of how the Speech-to-Speech project executes, following the actual code flow from startup to shutdown. Each code block is explained in the order it runs, making it accessible for junior Python developers to understand the entire system.

## 1. Application Entry Point (`s2s_pipeline.py`)

### 1.1 Initial Imports and Setup

```python
import logging
import os
import sys
from copy import copy
from pathlib import Path
from queue import Queue
from threading import Event
from typing import Optional
from sys import platform
```

**What happens here:**
- Python loads all necessary standard library modules
- `queue.Queue` will be used for thread-safe communication between pipeline stages
- `threading.Event` will coordinate between different threads
- `pathlib.Path` provides modern file path handling
- `sys.platform` helps detect if we're running on macOS, Windows, or Linux

```python
from VAD.vad_handler import VADHandler
from arguments_classes.chat_tts_arguments import ChatTTSHandlerArguments
# ... more imports
```

**What happens here:**
- Python imports all the custom classes we've defined
- Each import represents a different component of our pipeline
- The `arguments_classes` imports bring in configuration dataclasses
- Handler imports bring in the actual processing components

### 1.2 NLTK Resource Download

```python
try:
    nltk.data.find("tokenizers/punkt_tab")
except (LookupError, OSError):
    nltk.download("punkt_tab")
try:
    nltk.data.find("tokenizers/averaged_perceptron_tagger_eng")
except (LookupError, OSError):
    nltk.download("averaged_perceptron_tagger_eng")
```

**What happens here:**
- The app checks if NLTK (Natural Language Toolkit) resources are available
- `punkt_tab` is needed for sentence tokenization (splitting text into sentences)
- If these resources aren't found, it automatically downloads them
- This ensures the language model can properly split its responses into sentences for streaming

### 1.3 PyTorch Compilation Cache Setup

```python
CURRENT_DIR = Path(__file__).resolve().parent
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(CURRENT_DIR, "tmp")
```

**What happens here:**
- Gets the directory where the main script is located
- Sets up a cache directory for PyTorch's compilation system
- This speeds up subsequent runs by ~50% because compiled models are cached
- The cache stores optimized GPU kernels so they don't need to be recompiled

### 1.4 Main Function Entry

```python
def main():
    (
        module_kwargs,
        socket_receiver_kwargs,
        socket_sender_kwargs,
        # ... more kwargs
    ) = parse_arguments()
```

**What happens here:**
- `main()` is called when the script runs
- `parse_arguments()` reads command-line arguments or JSON config files
- Returns separate configuration objects for each component
- Each `*_kwargs` object contains settings for a specific part of the pipeline

## 2. Argument Parsing (`parse_arguments()`)

### 2.1 HuggingFace Argument Parser Setup

```python
def parse_arguments():
    parser = HfArgumentParser(
        (
            ModuleArguments,
            SocketReceiverArguments,
            SocketSenderArguments,
            # ... all argument classes
        )
    )
```

**What happens here:**
- Creates a parser that understands dataclass-based arguments
- Each argument class defines configuration for a specific component
- HuggingFace's parser automatically generates help text and validation
- Supports both command-line args and JSON configuration files

### 2.2 Parsing Logic

```python
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    return parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    return parser.parse_args_into_dataclasses()
```

**What happens here:**
- If user provides a JSON file, load configuration from there
- Otherwise, parse command-line arguments
- Returns a tuple of dataclass instances, each containing validated settings
- This flexibility allows both quick command-line usage and complex configuration files

## 3. Logging Setup (`setup_logger()`)

```python
def setup_logger(log_level):
    global logger
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
```

**What happens here:**
- Configures Python's logging system with timestamps and structured format
- Sets the global log level (DEBUG, INFO, WARNING, ERROR)
- Creates a logger instance for this module
- All subsequent log messages will follow this format

```python
if log_level == "debug":
    torch._logging.set_logs(graph_breaks=True, recompiles=True, cudagraphs=True)
```

**What happens here:**
- If debug mode is enabled, also turn on PyTorch's internal logging
- This shows when models are being compiled and optimized
- Helps developers understand performance characteristics
- Shows CUDA graph captures and compilation events

## 4. Configuration Processing

### 4.1 Mac Optimization Settings

```python
def optimal_mac_settings(mac_optimal_settings: Optional[str], *handler_kwargs):
    if mac_optimal_settings:
        for kwargs in handler_kwargs:
            if hasattr(kwargs, "device"):
                kwargs.device = "mps"  # Metal Performance Shaders
            if hasattr(kwargs, "stt"):
                kwargs.stt = "moonshine"  # Lightweight STT
            if hasattr(kwargs, "llm"):
                kwargs.llm = "mlx-lm"  # Apple MLX framework
            if hasattr(kwargs, "tts"):
                kwargs.tts = "melo"  # Efficient TTS
```

**What happens here:**
- If user enables Mac optimization, automatically configure best settings
- `mps` device uses Apple's GPU acceleration
- `moonshine` is a fast STT model optimized for Apple Silicon
- `mlx-lm` uses Apple's MLX framework for efficient inference
- `melo` TTS works well on Mac hardware
- This saves users from having to know the optimal configuration

### 4.2 Device Override

```python
def overwrite_device_argument(common_device: Optional[str], *handler_kwargs):
    if common_device:
        for kwargs in handler_kwargs:
            if hasattr(kwargs, "lm_device"):
                kwargs.lm_device = common_device
            if hasattr(kwargs, "tts_device"):
                kwargs.tts_device = common_device
            # ... more device assignments
```

**What happens here:**
- If user specifies a global device (like "cuda" or "cpu"), apply it everywhere
- Overrides individual device settings for each component
- Ensures all models run on the same device for consistency
- Prevents memory transfer overhead between devices

### 4.3 Argument Renaming

```python
def rename_args(args, prefix):
    gen_kwargs = {}
    for key in copy(args.__dict__):
        if key.startswith(prefix):
            value = args.__dict__.pop(key)
            new_key = key[len(prefix) + 1:]  # Remove prefix and underscore
            if new_key.startswith("gen_"):
                gen_kwargs[new_key[4:]] = value  # Remove 'gen_' and add to dict
            else:
                args.__dict__[new_key] = value
    args.__dict__["gen_kwargs"] = gen_kwargs
```

**What happens here:**
- Command-line args come with prefixes like `stt_model_name`, `lm_device`, etc.
- This function removes prefixes so handlers get clean parameter names
- Arguments starting with `gen_` become generation parameters
- For example: `stt_gen_max_new_tokens` becomes `gen_kwargs["max_new_tokens"]`
- This allows handlers to receive parameters in the format they expect

## 5. Queue and Event Initialization

```python
def initialize_queues_and_events():
    return {
        "stop_event": Event(),
        "should_listen": Event(),
        "recv_audio_chunks_queue": Queue(),
        "send_audio_chunks_queue": Queue(),
        "spoken_prompt_queue": Queue(),
        "text_prompt_queue": Queue(),
        "lm_response_queue": Queue(),
    }
```

**What happens here:**
- Creates all the communication channels between pipeline components
- `stop_event`: Signals all threads to shut down gracefully
- `should_listen`: Coordinates when the system should accept new audio input
- Each queue represents data flowing between specific pipeline stages:
  - `recv_audio_chunks_queue`: Raw audio from microphone/client
  - `spoken_prompt_queue`: Audio segments detected by VAD
  - `text_prompt_queue`: Transcribed text from STT
  - `lm_response_queue`: Generated text responses from LLM
  - `send_audio_chunks_queue`: Synthesized audio going to speakers/client

**Data Flow:**
```
Audio Input → recv_audio_chunks_queue → VAD → spoken_prompt_queue → 
STT → text_prompt_queue → LLM → lm_response_queue → TTS → send_audio_chunks_queue → Audio Output
```

## 6. Pipeline Construction (`build_pipeline()`)

### 6.1 Communication Handler Setup

```python
if module_kwargs.mode == "local":
    from connections.local_audio_streamer import LocalAudioStreamer
    local_audio_streamer = LocalAudioStreamer(
        input_queue=recv_audio_chunks_queue, 
        output_queue=send_audio_chunks_queue
    )
    comms_handlers = [local_audio_streamer]
    should_listen.set()
else:
    from connections.socket_receiver import SocketReceiver
    from connections.socket_sender import SocketSender
    comms_handlers = [
        SocketReceiver(stop_event, recv_audio_chunks_queue, should_listen, ...),
        SocketSender(stop_event, send_audio_chunks_queue, ...)
    ]
```

**What happens here:**
- Based on the mode setting, create appropriate communication handlers
- **Local mode**: Direct audio I/O using sounddevice library
- **Socket mode**: Network communication with separate receiver/sender
- `should_listen.set()` immediately enables audio input in local mode
- In socket mode, listening is controlled by the connection state

### 6.2 VAD Handler Creation

```python
vad = VADHandler(
    stop_event,
    queue_in=recv_audio_chunks_queue,
    queue_out=spoken_prompt_queue,
    setup_args=(should_listen,),
    setup_kwargs=vars(vad_handler_kwargs),
)
```

**What happens here:**
- Creates the Voice Activity Detection handler
- `queue_in`: Receives raw audio chunks
- `queue_out`: Sends detected speech segments
- `setup_args`: Passes the `should_listen` event for coordination
- `setup_kwargs`: Configuration like thresholds, timing parameters
- VAD will process audio continuously but only output when speech is detected

### 6.3 STT Handler Selection

```python
def get_stt_handler(module_kwargs, stop_event, spoken_prompt_queue, text_prompt_queue, ...):
    if module_kwargs.stt == "moonshine":
        from STT.moonshine_handler import MoonshineSTTHandler
        return MoonshineSTTHandler(stop_event, queue_in=spoken_prompt_queue, queue_out=text_prompt_queue)
    elif module_kwargs.stt == "whisper":
        from STT.whisper_stt_handler import WhisperSTTHandler
        return WhisperSTTHandler(stop_event, queue_in=spoken_prompt_queue, queue_out=text_prompt_queue, 
                               setup_kwargs=vars(whisper_stt_handler_kwargs))
    # ... more options
```

**What happens here:**
- Factory pattern for creating the correct STT handler
- Each STT implementation has the same interface but different capabilities
- Configuration is passed through `setup_kwargs`
- The handler will read from `spoken_prompt_queue` and write to `text_prompt_queue`

### 6.4 LLM and TTS Handler Creation

Similar patterns are used for LLM and TTS handlers:
- Factory functions select the correct implementation
- All handlers follow the same BaseHandler interface
- Queues connect the handlers in a pipeline
- Configuration is passed through setup parameters

## 7. Thread Manager Initialization

```python
return ThreadManager([*comms_handlers, vad, stt, lm, tts])
```

**What happens here:**
- Creates a ThreadManager with all pipeline components
- Each handler will run in its own thread
- ThreadManager coordinates startup and shutdown
- The `*` operator unpacks the communication handlers list

## 8. Pipeline Execution

### 8.1 Thread Startup

```python
def start(self):
    for handler in self.handlers:
        thread = threading.Thread(target=handler.run)
        self.threads.append(thread)
        thread.start()
```

**What happens here:**
- Creates a separate thread for each handler
- Each thread runs the handler's `run()` method
- Threads start immediately and begin processing
- All handlers now run concurrently

### 8.2 BaseHandler Execution Loop

```python
def run(self):
    while not self.stop_event.is_set():
        input = self.queue_in.get()  # Blocks until data available
        if isinstance(input, bytes) and input == b"END":
            logger.debug("Stopping thread")
            break
        start_time = perf_counter()
        for output in self.process(input):  # Handler-specific processing
            self._times.append(perf_counter() - start_time)
            if self.last_time > self.min_time_to_debug:
                logger.debug(f"{self.__class__.__name__}: {self.last_time: .3f} s")
            self.queue_out.put(output)
            start_time = perf_counter()
    
    self.cleanup()
    self.queue_out.put(b"END")
```

**What happens here:**
- Main execution loop for every handler
- `queue_in.get()` blocks the thread until new data arrives
- `b"END"` is a special signal that means "shutdown gracefully"
- Each handler implements its own `process()` method
- Performance timing tracks how long each processing step takes
- Results are put into the output queue for the next handler
- When stopping, cleanup is performed and `b"END"` is forwarded

## 9. Detailed Handler Execution

### 9.1 VAD Handler Processing

```python
def process(self, audio_chunk):
    audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
    audio_float32 = int2float(audio_int16)
    vad_output = self.iterator(torch.from_numpy(audio_float32))
    if vad_output is not None and len(vad_output) != 0:
        logger.debug("VAD: end of speech detected")
        array = torch.cat(vad_output).cpu().numpy()
        duration_ms = len(array) / self.sample_rate * 1000
        if duration_ms < self.min_speech_ms or duration_ms > self.max_speech_ms:
            logger.debug(f"audio input of duration: {len(array) / self.sample_rate}s, skipping")
        else:
            self.should_listen.clear()
            logger.debug("Stop listening")
            # ... optional audio enhancement
            yield array
```

**What happens here:**
1. **Audio Conversion**: Raw bytes → int16 numpy array → float32 for processing
2. **VAD Processing**: Silero VAD model analyzes the audio chunk
3. **Speech Detection**: If speech boundary is detected, VAD returns accumulated audio
4. **Duration Filtering**: Rejects audio that's too short or too long
5. **Listening Control**: `should_listen.clear()` prevents new audio input during processing
6. **Audio Enhancement**: Optional noise reduction if enabled
7. **Output**: Yields the speech segment to the STT handler

### 9.2 Whisper STT Processing

```python
def process(self, spoken_prompt):
    logger.debug("infering whisper...")
    global pipeline_start
    pipeline_start = perf_counter()
    
    input_features = self.prepare_model_inputs(spoken_prompt)
    pred_ids = self.model.generate(input_features, **self.gen_kwargs)
    language_code = self.processor.tokenizer.decode(pred_ids[0, 1])[2:-2]
    
    if language_code not in SUPPORTED_LANGUAGES:
        logger.warning("Whisper detected unsupported language:", language_code)
        gen_kwargs = copy(self.gen_kwargs)
        gen_kwargs['language'] = self.last_language
        language_code = self.last_language
        pred_ids = self.model.generate(input_features, **gen_kwargs)
    else:
        self.last_language = language_code
    
    pred_text = self.processor.batch_decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=False)[0]
    console.print(f"[yellow]USER: {pred_text}")
    
    if self.start_language == "auto":
        language_code += "-auto"
    
    yield (pred_text, language_code)
```

**What happens here:**
1. **Timing Start**: Records when the pipeline processing begins
2. **Input Preparation**: Converts audio to mel-spectrogram features
3. **Model Inference**: Whisper generates token predictions
4. **Language Detection**: Extracts language code from the first predicted token
5. **Language Validation**: Falls back to last known language if unsupported
6. **Text Decoding**: Converts tokens back to readable text
7. **User Feedback**: Displays transcribed text in yellow
8. **Language Tagging**: Adds "-auto" suffix for auto-detection mode
9. **Output**: Yields (text, language_code) tuple to LLM handler

### 9.3 Language Model Processing

```python
def process(self, prompt):
    logger.debug("infering language model...")
    language_code = None
    if isinstance(prompt, tuple):
        prompt, language_code = prompt
        if language_code[-5:] == "-auto":
            language_code = language_code[:-5]
            prompt = f"Please reply to my message in {WHISPER_LANGUAGE_TO_LLM_LANGUAGE[language_code]}. " + prompt

    self.chat.append({"role": self.user_role, "content": prompt})
    thread = Thread(target=self.pipe, args=(self.chat.to_list(),), kwargs=self.gen_kwargs)
    thread.start()
    
    generated_text, printable_text = "", ""
    for new_text in self.streamer:
        generated_text += new_text
        printable_text += new_text
        sentences = sent_tokenize(printable_text)
        if len(sentences) > 1:
            yield (sentences[0], language_code)
            printable_text = new_text

    self.chat.append({"role": "assistant", "content": generated_text})
    yield (printable_text, language_code)  # Don't forget last sentence
```

**What happens here:**
1. **Input Parsing**: Extracts text and language code from STT output
2. **Language Prompting**: If auto-detection, asks LLM to respond in detected language
3. **Chat Management**: Adds user message to conversation history
4. **Threaded Generation**: Runs model inference in separate thread for streaming
5. **Streaming Loop**: Processes each generated token as it arrives
6. **Sentence Detection**: Uses NLTK to identify complete sentences
7. **Early Yielding**: Sends complete sentences to TTS immediately (reduces latency)
8. **History Update**: Saves complete response to conversation memory
9. **Final Output**: Ensures the last sentence is sent even if incomplete

### 9.4 Parler TTS Processing

```python
def process(self, llm_sentence):
    if isinstance(llm_sentence, tuple):
        llm_sentence, language_code = llm_sentence
        self.speaker = WHISPER_LANGUAGE_TO_PARLER_SPEAKER.get(language_code, "Jason")
        
    console.print(f"[green]ASSISTANT: {llm_sentence}")
    nb_tokens = len(self.prompt_tokenizer(llm_sentence).input_ids)
    
    pad_args = {}
    if self.compile_mode:
        pad_length = next_power_of_2(nb_tokens)
        logger.debug(f"padding to {pad_length}")
        pad_args["pad"] = True
        pad_args["max_length_prompt"] = pad_length

    tts_gen_kwargs = self.prepare_model_inputs(llm_sentence, **pad_args)
    
    streamer = ParlerTTSStreamer(self.model, device=self.device, play_steps=self.play_steps)
    tts_gen_kwargs = {"streamer": streamer, **tts_gen_kwargs}
    torch.manual_seed(0)  # Deterministic audio generation
    thread = Thread(target=self.model.generate, kwargs=tts_gen_kwargs)
    thread.start()

    for i, audio_chunk in enumerate(streamer):
        global pipeline_start
        if i == 0 and "pipeline_start" in globals():
            logger.info(f"Time to first audio: {perf_counter() - pipeline_start:.3f}")
        audio_chunk = librosa.resample(audio_chunk, orig_sr=44100, target_sr=16000)
        audio_chunk = (audio_chunk * 32768).astype(np.int16)
        for i in range(0, len(audio_chunk), self.blocksize):
            yield np.pad(audio_chunk[i : i + self.blocksize], 
                        (0, self.blocksize - len(audio_chunk[i : i + self.blocksize])))

    self.should_listen.set()  # Re-enable audio input
```

**What happens here:**
1. **Speaker Selection**: Choose voice based on detected language
2. **User Feedback**: Display assistant response in green
3. **Token Counting**: Determine input length for optimization
4. **Padding Logic**: For compiled models, pad to power-of-2 for efficiency
5. **Model Preparation**: Set up generation parameters and voice description
6. **Streaming Setup**: Create audio streamer for real-time output
7. **Deterministic Generation**: Fixed random seed for consistent voice
8. **Threaded Generation**: Run TTS in separate thread
9. **Audio Processing**: Convert from 44.1kHz to 16kHz and normalize
10. **Chunking**: Split audio into fixed-size blocks for smooth playback
11. **Timing Measurement**: Track time from speech input to first audio output
12. **Re-enable Input**: Allow new speech input once TTS starts

## 10. Communication Handlers

### 10.1 Local Audio Streamer

```python
def callback(indata, outdata, frames, time, status):
    if self.output_queue.empty():
        self.input_queue.put(indata.copy())
        outdata[:] = 0 * outdata
    else:
        outdata[:] = self.output_queue.get()[:, np.newaxis]

with sd.Stream(samplerate=16000, dtype="int16", channels=1, 
               callback=callback, blocksize=self.list_play_chunk_size):
    logger.info("Starting local audio stream")
    while not self.stop_event.is_set():
        time.sleep(0.001)
```

**What happens here:**
1. **Audio Callback**: Called by sounddevice for each audio buffer
2. **Priority Logic**: If audio is ready to play, play it; otherwise record input
3. **Input Recording**: Copies microphone data to input queue
4. **Output Playback**: Retrieves synthesized audio from output queue
5. **Silence Generation**: Outputs zeros when no audio is available
6. **Stream Management**: Maintains audio stream until stop signal
7. **Low Latency**: 1ms sleep prevents busy waiting while maintaining responsiveness

### 10.2 Socket Communication

**Socket Receiver:**
```python
def receive_full_chunk(self, conn, chunk_size):
    data = b""
    while len(data) < chunk_size:
        packet = conn.recv(chunk_size - len(data))
        if not packet:
            return None  # Connection closed
        data += packet
    return data

def run(self):
    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.socket.bind((self.host, self.port))
    self.socket.listen(1)
    self.conn, _ = self.socket.accept()
    
    self.should_listen.set()
    while not self.stop_event.is_set():
        audio_chunk = self.receive_full_chunk(self.conn, self.chunk_size)
        if audio_chunk is None:
            self.queue_out.put(b"END")
            break
        if self.should_listen.is_set():
            self.queue_out.put(audio_chunk)
```

**What happens here:**
1. **Reliable Reception**: Ensures complete chunks are received (TCP can split packets)
2. **Connection Handling**: Waits for client connection, then maintains it
3. **Flow Control**: Only processes audio when `should_listen` is set
4. **Graceful Shutdown**: Detects connection loss and signals pipeline to stop
5. **Threading**: Runs in background thread to avoid blocking other components

## 11. Client Application (`listen_and_play.py`)

### 11.1 Audio Callbacks

```python
def callback_recv(outdata, frames, time, status):
    if not recv_queue.empty():
        data = recv_queue.get()
        outdata[: len(data)] = data
        outdata[len(data) :] = b"\x00" * (len(outdata) - len(data))
    else:
        outdata[:] = b"\x00" * len(outdata)

def callback_send(indata, frames, time, status):
    if recv_queue.empty():  # Only record when not playing
        data = bytes(indata)
        send_queue.put(data)
```

**What happens here:**
- **Receive Callback**: Plays audio from server, fills with silence if none available
- **Send Callback**: Records microphone input only when not receiving audio
- **Half-Duplex Logic**: Prevents feedback by not recording during playback
- **Queue Management**: Uses thread-safe queues for audio buffering

### 11.2 Network Threads

```python
def send(stop_event, send_queue):
    while not stop_event.is_set():
        data = send_queue.get()
        send_socket.sendall(data)

def recv(stop_event, recv_queue):
    while not stop_event.is_set():
        data = receive_full_chunk(recv_socket, list_play_chunk_size * 2)
        if data:
            recv_queue.put(data)
```

**What happens here:**
- **Send Thread**: Continuously sends microphone data to server
- **Receive Thread**: Continuously receives synthesized audio from server
- **Blocking Operations**: Both threads block on I/O operations
- **Error Handling**: Connection issues trigger graceful shutdown

## 12. Shutdown Sequence

### 12.1 Keyboard Interrupt Handling

```python
try:
    pipeline_manager.start()
except KeyboardInterrupt:
    pipeline_manager.stop()
```

**What happens here:**
- User presses Ctrl+C to interrupt the program
- Python raises KeyboardInterrupt exception
- This triggers the graceful shutdown process

### 12.2 ThreadManager Shutdown

```python
def stop(self):
    for handler in self.handlers:
        handler.stop_event.set()
    for thread in self.threads:
        thread.join()
```

**What happens here:**
1. **Signal All Handlers**: Sets stop_event for every handler
2. **Thread Joining**: Waits for each thread to finish its current work
3. **Coordinated Shutdown**: Ensures no data is lost during shutdown
4. **Resource Cleanup**: Each handler performs its own cleanup

### 12.3 Handler Cleanup

```python
def cleanup(self):
    print("Stopping FasterWhisperSTTHandler")
    del self.model
```

**What happens here:**
- Each handler implements its own cleanup logic
- Models are explicitly deleted to free GPU memory
- File handles and network connections are closed
- Resources are returned to the system

## Summary of Execution Flow

1. **Startup**: Parse arguments, configure logging, initialize queues
2. **Pipeline Building**: Create and configure all handlers based on settings
3. **Thread Launch**: Start all handlers in separate threads
4. **Audio Input**: Microphone/client provides audio chunks
5. **VAD Processing**: Detect speech boundaries and accumulate audio
6. **STT Processing**: Convert speech to text with language detection
7. **LLM Processing**: Generate response with streaming sentence output
8. **TTS Processing**: Synthesize speech with real-time audio streaming
9. **Audio Output**: Play synthesized audio through speakers/client
10. **Shutdown**: Graceful cleanup of all resources and threads

The system is designed for **real-time operation** with **minimal latency**. Each component processes data as soon as it's available, and streaming techniques ensure audio starts playing before text generation is complete. The modular architecture allows easy swapping of components while maintaining the same execution flow.

This design enables **natural conversation** where users can speak, receive immediate transcription feedback, and hear responses with minimal delay, creating a smooth interactive experience similar to talking with another person.