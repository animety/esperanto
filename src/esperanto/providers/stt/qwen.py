"""Qwen speech-to-text provider implementation.

This provider supports Alibaba Cloud's Qwen3-ASR models through DashScope API.
"""

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

import httpx

from esperanto.common_types import TranscriptionResponse
from esperanto.providers.stt.base import Model, SpeechToTextModel


@dataclass
class QwenSpeechToTextModel(SpeechToTextModel):
    """Qwen speech-to-text model implementation.
    
    Supports Qwen3-ASR-Flash (short audio, ≤5 min) and Qwen3-ASR-Flash-Filetrans (long audio, ≤12 hours).
    
    Features:
    - Automatic language detection (11 languages supported)
    - Context enhancement via system messages
    - High accuracy for Chinese and other languages
    - Support for URL, file path, and Base64 inputs
    
    Environment Variables:
        DASHSCOPE_API_KEY: API key for Alibaba Cloud DashScope
        QWEN_API_KEY: Alternative API key name (fallback)
    """

    def __post_init__(self):
        """Initialize HTTP clients and configuration."""
        # Call parent's post_init to handle config initialization
        super().__post_init__()

        # Get API key from environment
        self.api_key = self.api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Qwen API key not found. Set DASHSCOPE_API_KEY or QWEN_API_KEY environment variable."
            )

        # Set base URL (Beijing region by default)
        self.base_url = self.base_url or "https://dashscope.aliyuncs.com/api/v1"

        # Initialize HTTP clients with configurable timeout
        self._create_http_clients()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Qwen API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_message = error_data.get("message", f"HTTP {response.status_code}")
                error_code = error_data.get("code", "Unknown")
                raise RuntimeError(f"Qwen API error [{error_code}]: {error_message}")
            except RuntimeError:
                raise
            except Exception:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return "qwen3-asr-flash"

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "qwen"

    def _get_models(self) -> List[Model]:
        """List all available Qwen ASR models.
        
        Returns:
            List[Model]: Available Qwen ASR models
        """
        return [
            Model(
                id="qwen3-asr-flash",
                owned_by="alibaba",
                context_window=None,  # Audio models don't have context windows
            ),
            Model(
                id="qwen3-asr-flash-filetrans",
                owned_by="alibaba",
                context_window=None,
            ),
        ]

    def _prepare_audio_input(self, audio_file: Union[str, BinaryIO]) -> str:
        """Prepare audio input for Qwen API.
        
        Args:
            audio_file: Audio file path or BinaryIO object
            
        Returns:
            str: Base64-encoded audio data or URL
        """
        if isinstance(audio_file, str):
            # Check if it's a URL
            if audio_file.startswith(("http://", "https://")):
                return audio_file
            
            # Otherwise, treat as file path and encode to Base64
            with open(audio_file, "rb") as f:
                audio_bytes = f.read()
                return base64.b64encode(audio_bytes).decode('utf-8')
        else:
            # For BinaryIO, read and encode to Base64
            audio_bytes = audio_file.read()
            # Reset file position if possible
            if hasattr(audio_file, 'seek'):
                audio_file.seek(0)
            return base64.b64encode(audio_bytes).decode('utf-8')

    def _build_messages(
        self, 
        audio_input: str, 
        prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Build messages for Qwen ASR API.
        
        Args:
            audio_input: Base64-encoded audio or URL
            prompt: Optional context hint for better accuracy
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # Add system message with context hint if provided
        if prompt:
            messages.append({
                "role": "system",
                "content": [{"text": prompt}]
            })
        
        # Add user message with audio
        messages.append({
            "role": "user",
            "content": [{"audio": audio_input}]
        })
        
        return messages

    def transcribe(
        self,
        audio_file: Union[str, BinaryIO],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResponse:
        """Transcribe audio to text using Qwen3-ASR.
        
        Args:
            audio_file: Path to audio file, URL, or BinaryIO object
            language: Optional language code (e.g., 'zh', 'en'). 
                     If not provided, automatic detection is used.
            prompt: Optional context hint to improve accuracy (Qwen-specific feature)
            
        Returns:
            TranscriptionResponse containing the transcribed text and metadata
            
        Raises:
            RuntimeError: If transcription fails
        """
        try:
            # Prepare audio input
            audio_input = self._prepare_audio_input(audio_file)
            
            # Build messages
            messages = self._build_messages(audio_input, prompt)
            
            # Prepare request payload
            payload: Dict[str, Any] = {
                "model": self.get_model_name(),
                "input": {
                    "messages": messages
                }
            }
            
            # Add ASR options if language is specified
            if language:
                payload["parameters"] = {
                    "asr_options": {
                        "language": language,
                        "enable_itn": False  # Inverse Text Normalization
                    }
                }
            
            # Make API request
            response = self.client.post(
                f"{self.base_url}/services/aigc/multimodal-generation/generation",
                headers=self._get_headers(),
                json=payload
            )
            
            self._handle_error(response)
            response_data = response.json()
            
            # Extract transcription text
            # Qwen3-ASR returns: output.choices[0].message.content[0].text
            output = response_data.get("output", {})
            choices = output.get("choices", [])
            
            if not choices:
                raise RuntimeError("No transcription result returned from Qwen API")
            
            content = choices[0].get("message", {}).get("content", [])
            if not content or not isinstance(content, list):
                raise RuntimeError("Invalid transcription format from Qwen API")
            
            text = content[0].get("text", "")
            
            return TranscriptionResponse(
                text=text,
                language=language,  # Return the specified or detected language
                model=self.get_model_name(),
            )
            
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to transcribe audio: {str(e)}") from e

    async def atranscribe(
        self,
        audio_file: Union[str, BinaryIO],
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResponse:
        """Async transcribe audio to text using Qwen3-ASR.
        
        Args:
            audio_file: Path to audio file, URL, or BinaryIO object
            language: Optional language code (e.g., 'zh', 'en'). 
                     If not provided, automatic detection is used.
            prompt: Optional context hint to improve accuracy (Qwen-specific feature)
            
        Returns:
            TranscriptionResponse containing the transcribed text and metadata
            
        Raises:
            RuntimeError: If transcription fails
        """
        try:
            # Prepare audio input
            audio_input = self._prepare_audio_input(audio_file)
            
            # Build messages
            messages = self._build_messages(audio_input, prompt)
            
            # Prepare request payload
            payload: Dict[str, Any] = {
                "model": self.get_model_name(),
                "input": {
                    "messages": messages
                }
            }
            
            # Add ASR options if language is specified
            if language:
                payload["parameters"] = {
                    "asr_options": {
                        "language": language,
                        "enable_itn": False  # Inverse Text Normalization
                    }
                }
            
            # Make API request
            response = await self.async_client.post(
                f"{self.base_url}/services/aigc/multimodal-generation/generation",
                headers=self._get_headers(),
                json=payload
            )
            
            self._handle_error(response)
            response_data = response.json()
            
            # Extract transcription text
            # Qwen3-ASR returns: output.choices[0].message.content[0].text
            output = response_data.get("output", {})
            choices = output.get("choices", [])
            
            if not choices:
                raise RuntimeError("No transcription result returned from Qwen API")
            
            content = choices[0].get("message", {}).get("content", [])
            if not content or not isinstance(content, list):
                raise RuntimeError("Invalid transcription format from Qwen API")
            
            text = content[0].get("text", "")
            
            return TranscriptionResponse(
                text=text,
                language=language,  # Return the specified or detected language
                model=self.get_model_name(),
            )
            
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to transcribe audio: {str(e)}") from e
