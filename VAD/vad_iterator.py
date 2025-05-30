import torch


class VADIterator:
    def __init__(
        self,
        model,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ):
        """
        Mainly taken from https://github.com/snakers4/silero-vad
        Class for stream imitation

        Parameters
        ----------
        model: preloaded .jit/.onnx silero VAD model

        threshold: float (default - 0.5)
            Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
            It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

        sampling_rate: int (default - 16000)
            Currently silero VAD models support 8000 and 16000 sample rates

        min_silence_duration_ms: int (default - 100 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before separating it

        speech_pad_ms: int (default - 30 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side
        """

        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.is_speaking = False
        self.buffer = []

        # Validate sample rate - Silero VAD only supports these rates
        if sampling_rate not in [8000, 16000]:
            raise ValueError(
                "VADIterator does not support sampling rates other than [8000, 16000]"
            )

        # Convert time durations to sample counts for processing
        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):
        """Reset all internal states for new audio stream."""
        self.model.reset_states()  # Reset VAD model's internal state
        self.triggered = False     # Currently detecting speech
        self.temp_end = 0         # Temporary marker for potential speech end
        self.current_sample = 0   # Current position in audio stream

    @torch.no_grad()  # Disable gradients for inference efficiency
    def __call__(self, x):
        """
        Process audio chunk and return complete speech segment when detected.
        
        Args:
            x: Audio chunk as torch.Tensor
            
        Returns:
            List of audio tensors representing complete speech segment, or None
        """
        # Ensure input is a tensor
        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except Exception:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        # Track current position in audio stream
        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        # Get speech probability from VAD model (0-1, higher = more likely speech)
        speech_prob = self.model(x, self.sampling_rate).item()

        # If speech detected during potential ending, cancel the ending
        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        # Start speech detection when threshold exceeded
        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            return None

        # End speech detection with hysteresis (threshold - 0.15 to avoid flickering)
        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                # Mark potential end of speech
                self.temp_end = self.current_sample
            
            # Check if we've had enough silence to confirm speech end
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                # end of speak
                self.temp_end = 0
                self.triggered = False
                spoken_utterance = self.buffer  # Get accumulated speech chunks
                self.buffer = []               # Clear buffer for next speech
                return spoken_utterance

        # Accumulate audio chunks while speech is active
        if self.triggered:
            self.buffer.append(x)

        return None
