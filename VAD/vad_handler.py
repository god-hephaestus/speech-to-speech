import torchaudio
from VAD.vad_iterator import VADIterator
from baseHandler import BaseHandler
import numpy as np
import torch
from rich.console import Console

from utils.utils import int2float
from df.enhance import enhance, init_df
import logging

logger = logging.getLogger(__name__)

console = Console()


class VADHandler(BaseHandler):
    """
    Handles voice activity detection. When voice activity is detected, audio will be accumulated until the end of speech is detected and then passed
    to the following part.
    """

    def setup(
        self,
        should_listen,
        thresh=0.3,
        sample_rate=16000,
        min_silence_ms=1000,
        min_speech_ms=500,
        max_speech_ms=float("inf"),
        speech_pad_ms=30,
        audio_enhancement=False,
    ):
        """
        Initialize the VAD handler with speech detection parameters and optional audio enhancement.
        
        Args:
            should_listen: Threading event to control listening state
            thresh: Speech detection threshold (0-1, lower = more sensitive)
            sample_rate: Audio sample rate in Hz
            min_silence_ms: Minimum silence duration before ending speech detection
            min_speech_ms: Minimum speech duration to be considered valid
            max_speech_ms: Maximum allowed speech duration
            speech_pad_ms: Padding around speech segments in milliseconds
            audio_enhancement: Whether to apply noise reduction/enhancement
        """
        # Store configuration parameters
        self.should_listen = should_listen
        self.sample_rate = sample_rate
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        
        # Load pre-trained Silero VAD model from PyTorch Hub
        self.model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
        
        # Create VAD iterator that handles the speech detection logic
        self.iterator = VADIterator(
            self.model,
            threshold=thresh,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
        # Setup audio enhancement if requested
        self.audio_enhancement = audio_enhancement
        if audio_enhancement:
            self.enhanced_model, self.df_state, _ = init_df()

    def process(self, audio_chunk):
        """
        Process incoming audio chunks and detect speech segments.
        
        Args:
            audio_chunk: Raw audio data as bytes
            
        Yields:
            numpy.ndarray: Complete speech segments when detected
        """
        # Convert raw audio bytes to numpy arrays
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)  # Raw bytes to 16-bit integers
        audio_float32 = int2float(audio_int16)  # Convert to 32-bit float for processing
        
        # Run VAD on the audio chunk - returns complete speech segment when speech ends
        vad_output = self.iterator(torch.from_numpy(audio_float32))
        if vad_output is not None and len(vad_output) != 0:
            logger.debug("VAD: end of speech detected")
            
            # Concatenate all buffered audio chunks into single array
            array = torch.cat(vad_output).cpu().numpy()
            duration_ms = len(array) / self.sample_rate * 1000
            
            # Filter out speech segments that are too short or too long
            if duration_ms < self.min_speech_ms or duration_ms > self.max_speech_ms:
                logger.debug(
                    f"audio input of duration: {len(array) / self.sample_rate}s, skipping"
                )
            else:
                # Valid speech detected - temporarily stop listening and process
                self.should_listen.clear()
                logger.debug("Stop listening")
                # Apply audio enhancement if enabled
                if self.audio_enhancement:
                    # Handle sample rate conversion for enhancement model
                    if self.sample_rate != self.df_state.sr():
                        # Resample to enhancement model's sample rate
                        audio_float32 = torchaudio.functional.resample(
                            torch.from_numpy(array),
                            orig_freq=self.sample_rate,
                            new_freq=self.df_state.sr(),
                        )
                        # Apply noise reduction/enhancement
                        enhanced = enhance(
                            self.enhanced_model,
                            self.df_state,
                            audio_float32.unsqueeze(0),  # Add batch dimension
                        )
                        # Resample back to original sample rate
                        enhanced = torchaudio.functional.resample(
                            enhanced,
                            orig_freq=self.df_state.sr(),
                            new_freq=self.sample_rate,
                        )
                    else:
                        # No resampling needed
                        enhanced = enhance(
                            self.enhanced_model, self.df_state, audio_float32
                        )
                    array = enhanced.numpy().squeeze()
                yield array

    @property
    def min_time_to_debug(self):
        return 0.00001
