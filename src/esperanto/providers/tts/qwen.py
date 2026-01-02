"""Qwen text-to-speech provider implementation.

This provider supports Alibaba Cloud's Qwen3-TTS models through DashScope API.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import httpx

from esperanto.common_types import Model
from esperanto.common_types.tts import AudioResponse, Voice
from esperanto.providers.tts.base import TextToSpeechModel


class QwenTextToSpeechModel(TextToSpeechModel):
    """Qwen text-to-speech model implementation.
    
    Supports Qwen3-TTS-Flash with 49 voice options and 10+ languages.
    
    Features:
    - 49 different voices (male, female, various styles)
    - Multi-language support (Chinese, English, Japanese, Korean, etc.)
    - Chinese dialect support (Beijing, Shanghai, Sichuan, Cantonese, etc.)
    - Fast inference with Qwen3-TTS-Flash model
    - High-quality, natural-sounding speech
    
    Environment Variables:
        DASHSCOPE_API_KEY: API key for Alibaba Cloud DashScope
        QWEN_API_KEY: Alternative API key name (fallback)
    """

    DEFAULT_MODEL = "qwen3-tts-flash"
    DEFAULT_VOICE = "Cherry"
    PROVIDER = "qwen"

    # Popular Qwen3-TTS voices (subset of 49 available voices)
    VOICE_DEFINITIONS = {
        # Female voices
        "Cherry": Voice(
            name="Cherry",
            id="Cherry",
            gender="FEMALE",
            language_code="zh-CN",
            description="Clear and professional female voice (Chinese/English)"
        ),
        "Jada": Voice(
            name="Jada",
            id="Jada",
            gender="FEMALE",
            language_code="zh-CN",
            description="Warm and friendly female voice (Chinese/English)"
        ),
        "Sunny": Voice(
            name="Sunny",
            id="Sunny",
            gender="FEMALE",
            language_code="zh-CN",
            description="Energetic and bright female voice (Chinese/English)"
        ),
        "Lisa": Voice(
            name="Lisa",
            id="Lisa",
            gender="FEMALE",
            language_code="zh-CN",
            description="Elegant and expressive female voice (Chinese/English)"
        ),
        
        # Male voices
        "Dylan": Voice(
            name="Dylan",
            id="Dylan",
            gender="MALE",
            language_code="zh-CN",
            description="Mature and authoritative male voice (Chinese/English)"
        ),
        "Leo": Voice(
            name="Leo",
            id="Leo",
            gender="MALE",
            language_code="zh-CN",
            description="Calm and steady male voice (Chinese/English)"
        ),
        "Alex": Voice(
            name="Alex",
            id="Alex",
            gender="MALE",
            language_code="zh-CN",
            description="Professional male voice for business content"
        ),
        
        # Dialect voices (Chinese)
        "Beijing": Voice(
            name="Beijing",
            id="Beijing",
            gender="FEMALE",
            language_code="zh-CN-beijing",
            description="Beijing dialect female voice"
        ),
        "Shanghai": Voice(
            name="Shanghai",
            id="Shanghai",
            gender="FEMALE",
            language_code="zh-CN-shanghai",
            description="Shanghai dialect female voice"
        ),
        "Sichuan": Voice(
            name="Sichuan",
            id="Sichuan",
            gender="FEMALE",
            language_code="zh-CN-sichuan",
            description="Sichuan dialect female voice"
        ),
        "Guangzhou": Voice(
            name="Guangzhou",
            id="Guangzhou",
            gender="FEMALE",
            language_code="zh-CN-guangzhou",
            description="Cantonese dialect female voice"
        ),
    }

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """Initialize Qwen TTS provider.
        
        Args:
            model_name: Name of the model to use (default: qwen3-tts-flash)
            api_key: Qwen API key. If not provided, will try DASHSCOPE_API_KEY or QWEN_API_KEY env var
            base_url: Optional base URL for the API (default: DashScope Beijing region)
            **kwargs: Additional configuration options
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY"),
            base_url=base_url,
            config=kwargs
        )

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

    @property
    def available_voices(self) -> Dict[str, Voice]:
        """Get available voices from Qwen TTS.

        Returns:
            Dict[str, Voice]: Dictionary of available voices with their information
        """
        return self.VOICE_DEFINITIONS

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return self.PROVIDER

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return self.DEFAULT_MODEL

    def _get_models(self) -> List[Model]:
        """List all available Qwen TTS models.
        
        Returns:
            List[Model]: Available Qwen TTS models
        """
        return [
            Model(
                id="qwen3-tts-flash",
                owned_by="alibaba",
                context_window=None,  # TTS models don't have context windows
            ),
            Model(
                id="qwen-tts",
                owned_by="alibaba",
                context_window=None,
            ),
        ]

    def _detect_language_type(self, text: str) -> str:
        """Detect language type from text for Qwen TTS.
        
        Args:
            text: Input text
            
        Returns:
            Language type string (e.g., "Chinese", "English")
        """
        # Simple heuristic: check if text contains Chinese characters
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
        has_japanese = any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text)
        has_korean = any('\uac00' <= char <= '\ud7af' for char in text)
        
        if has_chinese:
            return "Chinese"
        elif has_japanese:
            return "Japanese"
        elif has_korean:
            return "Korean"
        else:
            return "English"

    def _download_audio(self, audio_url: str) -> bytes:
        """Download audio from URL.
        
        Args:
            audio_url: URL to audio file
            
        Returns:
            bytes: Audio data
        """
        response = self.client.get(audio_url)
        response.raise_for_status()
        return response.content

    async def _adownload_audio(self, audio_url: str) -> bytes:
        """Async download audio from URL.
        
        Args:
            audio_url: URL to audio file
            
        Returns:
            bytes: Audio data
        """
        response = await self.async_client.get(audio_url)
        response.raise_for_status()
        return response.content

    def generate_speech(
        self,
        text: str,
        voice: str = DEFAULT_VOICE,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioResponse:
        """Generate speech from text using Qwen3-TTS.

        Args:
            text: Text to convert to speech (max 600 characters for qwen3-tts-flash)
            voice: Voice to use (default: "Cherry")
            output_file: Optional path to save the audio file
            **kwargs: Additional parameters:
                - language_type: Language type (auto-detected if not provided)
                - stream: Whether to use streaming (default: False)

        Returns:
            AudioResponse containing the audio data and metadata

        Raises:
            RuntimeError: If speech generation fails
        """
        try:
            # Validate input
            self.validate_parameters(text, voice, self.model_name)
            
            # Get language type (auto-detect or from kwargs)
            language_type = kwargs.pop("language_type", None)
            if not language_type:
                language_type = self._detect_language_type(text)
            
            # Prepare request payload
            # Qwen3-TTS API format (HTTP direct call)
            payload = {
                "model": self.model_name,
                "input": {
                    "text": text
                },
                "parameters": {
                    "voice": voice,
                    "text_type": "PlainText",
                    "format": "wav",
                    "sample_rate": 16000
                }
            }
            # Add language_type if not default
            if language_type:
                payload["parameters"]["language_type"] = language_type
            # Add any additional kwargs
            if kwargs:
                payload["parameters"].update(kwargs)

            # Generate speech
            response = self.client.post(
                f"{self.base_url}/services/aigc/multimodal-generation/generation",
                headers=self._get_headers(),
                json=payload
            )
            self._handle_error(response)
            
            response_data = response.json()
            
            # Extract audio URL from response
            # Qwen3-TTS returns: output.audio.url
            output = response_data.get("output", {})
            audio_info = output.get("audio", {})
            
            if not audio_info:
                raise RuntimeError("No audio data returned from Qwen API")
            
            audio_url = audio_info.get("url")
            if not audio_url:
                raise RuntimeError("No audio URL in Qwen API response")
            
            # Download audio from URL
            audio_data = self._download_audio(audio_url)
            
            # Save to file if specified
            if output_file:
                output_file = Path(output_file)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_bytes(audio_data)

            return AudioResponse(
                audio_data=audio_data,
                content_type="audio/wav",  # Qwen3-TTS returns WAV format
                model=self.model_name,
                voice=voice,
                provider=self.PROVIDER,
                metadata={
                    "text": text,
                    "language_type": language_type,
                    "audio_url": audio_url,
                    "audio_id": audio_info.get("id"),
                }
            )

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to generate speech: {str(e)}") from e

    async def agenerate_speech(
        self,
        text: str,
        voice: str = DEFAULT_VOICE,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioResponse:
        """Generate speech from text using Qwen3-TTS asynchronously.

        Args:
            text: Text to convert to speech (max 600 characters for qwen3-tts-flash)
            voice: Voice to use (default: "Cherry")
            output_file: Optional path to save the audio file
            **kwargs: Additional parameters:
                - language_type: Language type (auto-detected if not provided)
                - stream: Whether to use streaming (default: False)

        Returns:
            AudioResponse containing the audio data and metadata

        Raises:
            RuntimeError: If speech generation fails
        """
        try:
            # Validate input
            self.validate_parameters(text, voice, self.model_name)
            
            # Get language type (auto-detect or from kwargs)
            language_type = kwargs.pop("language_type", None)
            if not language_type:
                language_type = self._detect_language_type(text)
            
            # Prepare request payload
            # Qwen3-TTS API format (HTTP direct call)
            payload = {
                "model": self.model_name,
                "input": {
                    "text": text
                },
                "parameters": {
                    "voice": voice,
                    "text_type": "PlainText",
                    "format": "wav",
                    "sample_rate": 16000
                }
            }
            # Add language_type if not default
            if language_type:
                payload["parameters"]["language_type"] = language_type
            # Add any additional kwargs
            if kwargs:
                payload["parameters"].update(kwargs)

            # Generate speech
            response = await self.async_client.post(
                f"{self.base_url}/services/aigc/multimodal-generation/generation",
                headers=self._get_headers(),
                json=payload
            )
            self._handle_error(response)
            
            response_data = response.json()
            
            # Extract audio URL from response
            # Qwen3-TTS returns: output.audio.url
            output = response_data.get("output", {})
            audio_info = output.get("audio", {})
            
            if not audio_info:
                raise RuntimeError("No audio data returned from Qwen API")
            
            audio_url = audio_info.get("url")
            if not audio_url:
                raise RuntimeError("No audio URL in Qwen API response")
            
            # Download audio from URL
            audio_data = await self._adownload_audio(audio_url)
            
            # Save to file if specified
            if output_file:
                output_file = Path(output_file)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_bytes(audio_data)

            return AudioResponse(
                audio_data=audio_data,
                content_type="audio/wav",  # Qwen3-TTS returns WAV format
                model=self.model_name,
                voice=voice,
                provider=self.PROVIDER,
                metadata={
                    "text": text,
                    "language_type": language_type,
                    "audio_url": audio_url,
                    "audio_id": audio_info.get("id"),
                }
            )

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to generate speech: {str(e)}") from e
