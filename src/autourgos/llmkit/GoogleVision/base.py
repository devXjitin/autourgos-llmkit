"""
Google Gemini Vision LLM Provider

Production-ready wrapper around Google's Generative AI client for Multimodal inputs.

Author: Autourgos Developer
Version: 1.1.0
"""

from typing import Optional, Any, Dict, List, Union
import os
import time
import warnings
import sys
from contextlib import contextmanager

# ============================================================================
# Environment Configuration
# ============================================================================
# Suppress verbose logging from gRPC and underlying libraries
os.environ['GRPC_VERBOSITY'] = 'ERROR'          # Suppress gRPC verbose logs
os.environ['GLOG_minloglevel'] = '2'            # Suppress Google logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        # Suppress TensorFlow logs

# Suppress Python warnings from gRPC modules
warnings.filterwarnings('ignore', category=UserWarning, module='.*grpc.*')

@contextmanager
def suppress_stderr():
    """Temporarily suppress stderr output using low-level file descriptor redirection."""
    import io
    
    original_stderr = sys.stderr
    original_stderr_fd = None
    
    try:
        # Save and redirect file descriptor
        try:
            original_stderr_fd = os.dup(2)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 2)
            os.close(devnull)
        except Exception:
            pass
        
        # Redirect Python's sys.stderr
        sys.stderr = io.StringIO()
        yield
    finally:
        # Restore stderr
        if original_stderr_fd is not None:
            try:
                os.dup2(original_stderr_fd, 2)
                os.close(original_stderr_fd)
            except Exception:
                pass
        sys.stderr = original_stderr

# ============================================================================
# Module-Level Client Import
# ============================================================================
# Import Google generativeai client at module level for better performance
_GOOGLE_GENAI_AVAILABLE = False
genai_module = None
try:
    with suppress_stderr():
        try:
            # Preferred: standard packaging
            import google.generativeai as genai_module  # type: ignore
            _GOOGLE_GENAI_AVAILABLE = True
        except Exception:
            # Fallback: alternate or older packaging structure
            from google import genai as genai_module  # type: ignore
            _GOOGLE_GENAI_AVAILABLE = True
except Exception:
    _GOOGLE_GENAI_AVAILABLE = False
    genai_module = None


# ============================================================================
# Custom Exception Hierarchy
# ============================================================================
class GoogleLLMError(Exception):
    """
    Base exception class for all Google Gemini LLM-related errors.
    """


class GoogleLLMImportError(GoogleLLMError):
    """
    Raised when the Google generative AI client library cannot be imported or initialized.
    """


class GoogleLLMAPIError(GoogleLLMError):
    """
    Raised when the API request fails after all retry attempts.
    """


class GoogleLLMResponseError(GoogleLLMError):
    """
    Raised when the API response cannot be interpreted or is malformed.
    """


def _extract_text_from_response(resp: Any) -> Optional[str]:
    """
    Extract generated text from Google API response using defensive parsing.
    
    Args:
        resp: Response object from Google Gemini API
    
    Returns:
        Extracted text string if found, None if extraction fails
    """
    if resp is None:
        return None

    # Strategy 1: Direct .text attribute (most common)
    try:
        text = getattr(resp, "text", None)
        if text is not None:
            if callable(text):
                text = text()
            if isinstance(text, str) and text.strip():
                return text
    except Exception:
        pass

    # Strategy 2: Structured candidates format
    candidates = getattr(resp, "candidates", None)
    if candidates and isinstance(candidates, (list, tuple)) and candidates:
        first = candidates[0]
        
        # Try candidate.content.parts[0].text
        content = getattr(first, "content", None)
        if content:
            parts = getattr(content, "parts", None)
            if parts and isinstance(parts, (list, tuple)) and parts:
                part_text = getattr(parts[0], "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    return part_text
            
            # Try content.text
            content_text = getattr(content, "text", None)
            if isinstance(content_text, str) and content_text.strip():
                return content_text
        
        # Try candidate.text
        candidate_text = getattr(first, "text", None)
        if isinstance(candidate_text, str) and candidate_text.strip():
            return candidate_text

    # Strategy 3: Dictionary-based responses
    if isinstance(resp, dict):
        candidates = resp.get("candidates")
        if candidates and isinstance(candidates, (list, tuple)) and candidates:
            first = candidates[0]
            if isinstance(first, dict):
                content = first.get("content")
                if isinstance(content, dict):
                    parts = content.get("parts")
                    if isinstance(parts, (list, tuple)) and parts:
                        part = parts[0]
                        if isinstance(part, dict):
                            text = part.get("text")
                            if isinstance(text, str) and text.strip():
                                return text
                
                # Try simpler structures
                text = first.get("content") or first.get("text")
                if isinstance(text, str) and text.strip():
                    return text
        
        # Try top-level text field
        text = resp.get("text")
        if isinstance(text, str) and text.strip():
            return text

    return None


def _extract_usage_from_response(resp: Any) -> dict:
    """
    Extract usage metadata (token counts) from Google API response.
    
    Args:
        resp: Response object from Google Gemini API
    
    Returns:
        Dictionary with token usage information
    """
    usage_data = {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }
    
    if resp is None:
        return usage_data
    
    # Strategy 1: Direct usage_metadata attribute
    try:
        usage = getattr(resp, "usage_metadata", None)
        if usage:
            usage_data["input_tokens"] = getattr(usage, "prompt_token_count", None)
            usage_data["output_tokens"] = getattr(usage, "candidates_token_count", None)
            usage_data["total_tokens"] = getattr(usage, "total_token_count", None)
            return usage_data
    except Exception:
        pass
    
    # Strategy 2: Dictionary-based usage metadata
    if isinstance(resp, dict):
        usage = resp.get("usage_metadata") or resp.get("usage")
        if isinstance(usage, dict):
            usage_data["input_tokens"] = usage.get("prompt_token_count") or usage.get("promptTokens")
            usage_data["output_tokens"] = usage.get("candidates_token_count") or usage.get("completionTokens")
            usage_data["total_tokens"] = usage.get("total_token_count") or usage.get("totalTokens")
    
    return usage_data


def _prepare_contents(prompt: str, images: Optional[Union[str, List[str], Any, List[Any]]] = None) -> List[Any]:
    """
    Prepare the content list for the Gemini API, handling text and images.
    
    Args:
        prompt: The text prompt.
        images: A single image or list of images. Can be file paths (str) or PIL Image objects.
        
    Returns:
        List of content items (text and image objects).
    """
    contents = [prompt]
    
    if not images:
        return contents
        
    if not isinstance(images, list):
        images = [images]
        
    for img in images:
        if isinstance(img, str):
            # Attempt to open image from path
            try:
                import PIL.Image
                image = PIL.Image.open(img)
                contents.append(image)
            except ImportError:
                 raise GoogleLLMImportError("Pillow (PIL) is required for image processing. Please install it: pip install Pillow")
            except Exception as e:
                raise ValueError(f"Failed to load image from path '{img}': {e}")
        else:
            # Assume it's a PIL Image or compatible object
            contents.append(img)
            
    return contents


def google_vision_llm(
    prompt: str,
    images: Optional[Union[str, List[str], Any, List[Any]]],
    model: str,
    api_key: Optional[str] = None,
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    timeout: Optional[float] = 30.0,
    backoff_factor: float = 0.5,
) -> dict:
    """Call a Google generative vision model and return structured response.

    Args:
        prompt: The prompt / input text.
        images: Image input(s). Can be a single path/object or list of paths/objects.
        model: Model identifier (e.g. "gemini-3-flash-preview", "gemini-3.1-pro-preview").
        api_key: API key. Defaults to GOOGLE_API_KEY env var.
        temperature: Controls randomness (0.0-2.0).
        top_p: Nucleus sampling threshold (0.0-1.0).
        top_k: Top-k sampling.
        max_tokens: Maximum tokens to generate.
        max_retries: Retry attempts.
        timeout: Request timeout.
        backoff_factor: Backoff factor.

    Returns:
        Dictionary with content, model, and usage.
    """

    # Input Validation
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    if not isinstance(max_retries, int) or max_retries < 1:
        raise ValueError("max_retries must be an integer >= 1")
    
    # Validate generation parameters
    if temperature is not None and not (0.0 <= temperature <= 2.0):
        raise ValueError("temperature must be between 0.0 and 2.0")
    if top_p is not None and not (0.0 <= top_p <= 1.0):
        raise ValueError("top_p must be between 0.0 and 1.0")
    
    # Build Generation Configuration
    generation_config = {}
    if temperature is not None:
        generation_config["temperature"] = temperature
    if top_p is not None:
        generation_config["top_p"] = top_p
    if top_k is not None:
        generation_config["top_k"] = top_k
    if max_tokens is not None:
        generation_config["max_output_tokens"] = max_tokens

    # API Key Configuration
    api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise GoogleLLMImportError(
            "No API key provided and environment variable GOOGLE_API_KEY is not set"
        )

    # Client Availability Check
    if not _GOOGLE_GENAI_AVAILABLE or genai_module is None:
        raise GoogleLLMImportError(
            "Failed to import or initialize google.generativeai client"
        )

    # Prepare Multimodal Content
    contents = _prepare_contents(prompt, images)

    # Client Configuration
    genai = genai_module
    client = None
    
    # Configure API key
    try:
        with suppress_stderr():
            cfg = getattr(genai, "configure", None)
            if callable(cfg):
                cfg(api_key=api_key)
    except Exception:
        pass

    # Initialize client object
    try:
        with suppress_stderr():
            ClientCls = getattr(genai, "Client", None)
            if callable(ClientCls):
                try:
                    client = ClientCls(api_key=api_key)
                except TypeError:
                    client = ClientCls()
    except Exception:
        pass

    # Retry Loop
    last_exc: Optional[BaseException] = None

    for attempt in range(1, max_retries + 1):
        try:
            with suppress_stderr():
                # Strategy 1: Client-based API
                if client is not None:
                    models_attr = getattr(client, "models", None)
                    gen_fn = getattr(models_attr, "generate_content", None) if models_attr else None
                    if callable(gen_fn):
                        kwargs: Dict[str, Any] = {}
                        if generation_config:
                            kwargs["config"] = generation_config
                        
                        resp = gen_fn(model=model, contents=contents, **kwargs)
                        text = _extract_text_from_response(resp)
                        if text:
                            usage = _extract_usage_from_response(resp)
                            return {
                                "content": text,
                                "model": model,
                                "usage": usage,
                            }

                # Strategy 2: GenerativeModel Class
                GenerativeModel = getattr(genai, "GenerativeModel", None)
                if callable(GenerativeModel):
                    try:
                        if generation_config:
                            model_obj = GenerativeModel(model, generation_config=generation_config)
                        else:
                            model_obj = GenerativeModel(model)
                        
                        gen_fn = getattr(model_obj, "generate_content", None)
                        if callable(gen_fn):
                            kwargs: Dict[str, Any] = {}
                            if timeout:
                                kwargs["request_options"] = {"timeout": timeout}
                            
                            resp = gen_fn(contents, **kwargs)
                            text = _extract_text_from_response(resp)
                            if text:
                                usage = _extract_usage_from_response(resp)
                                return {
                                    "content": text,
                                    "model": model,
                                    "usage": usage,
                                }
                    except Exception:
                        pass

                # Strategy 3: Top-level convenience functions (unlikely to handle multimodal well if simple 'generate_text', but 'generate' might)
                # We skip 'generate_text' as it is usually text-only legacy.

            # No valid response extracted
            raise GoogleLLMResponseError("No text could be extracted from the API response")

        except Exception as exc:
            last_exc = exc
            if attempt == max_retries:
                raise GoogleLLMAPIError(
                    f"Google Vision LLM request failed after {max_retries} attempts: {exc}"
                ) from exc

            time.sleep(backoff_factor * (2 ** (attempt - 1)))

    raise GoogleLLMAPIError("Google Vision LLM request failed") from last_exc


def google_vision_llm_stream(
    prompt: str,
    images: Optional[Union[str, List[str], Any, List[Any]]],
    model: str,
    api_key: Optional[str] = None,
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = 30.0,
):
    """Call a Google generative vision model and stream generated text chunks.

    Args:
        prompt: The prompt / input text.
        images: Image input(s).
        model: Model identifier.
        api_key: API key.
        temperature: Controls randomness.
        top_p: Nucleus sampling.
        top_k: Top-k sampling.
        max_tokens: Maximum tokens.
        timeout: Optional timeout.

    Yields:
        Text chunks as they are generated by the model.

    Example:
        >>> for chunk in google_vision_llm_stream(
        ...     prompt="Describe this image",
        ...     images="image.jpg",
        ...     model="gemini-3-flash-preview"
        ... ):
        ...     print(chunk, end='', flush=True)
    """

    # Input Validation
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    
    # Config validation...
    if temperature is not None and not (0.0 <= temperature <= 2.0):
        raise ValueError("temperature must be between 0.0 and 2.0")

    generation_config = {}
    if temperature is not None:
        generation_config["temperature"] = temperature
    if top_p is not None:
        generation_config["top_p"] = top_p
    if top_k is not None:
        generation_config["top_k"] = top_k
    if max_tokens is not None:
        generation_config["max_output_tokens"] = max_tokens

    api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise GoogleLLMImportError("No API key provided")

    if not _GOOGLE_GENAI_AVAILABLE or genai_module is None:
        raise GoogleLLMImportError("Failed to import google.generativeai client")

    contents = _prepare_contents(prompt, images)

    genai = genai_module
    client = None
    
    try:
        with suppress_stderr():
            cfg = getattr(genai, "configure", None)
            if callable(cfg):
                cfg(api_key=api_key)
    except Exception:
        pass

    try:
        with suppress_stderr():
            ClientCls = getattr(genai, "Client", None)
            if callable(ClientCls):
                try:
                    client = ClientCls(api_key=api_key)
                except TypeError:
                    client = ClientCls()
    except Exception:
        pass

    try:
        with suppress_stderr():
            GenerativeModel = getattr(genai, "GenerativeModel", None)
            if callable(GenerativeModel):
                try:
                    if generation_config:
                        model_obj = GenerativeModel(model, generation_config=generation_config)
                    else:
                        model_obj = GenerativeModel(model)
                    
                    gen_fn = getattr(model_obj, "generate_content", None)
                    if callable(gen_fn):
                        kwargs: Dict[str, Any] = {"stream": True}
                        if timeout:
                            kwargs["request_options"] = {"timeout": timeout}
                        
                        response_stream = gen_fn(contents, **kwargs)
                        
                        for chunk in response_stream:  # type: ignore
                            text = _extract_text_from_response(chunk)
                            if text:
                                yield text
                        return
                except Exception:
                    pass
            
            # Client based stream
            if client is not None:
                models_attr = getattr(client, "models", None)
                stream_fn = getattr(models_attr, "stream_generate_content", None) if models_attr else None
                if callable(stream_fn):
                    kwargs: Dict[str, Any] = {"timeout": timeout} if timeout else {}
                    if generation_config:
                        kwargs["config"] = generation_config
                        
                    response_stream = stream_fn(model=model, contents=contents, **kwargs)
                    for chunk in response_stream:  # type: ignore
                        text = _extract_text_from_response(chunk)
                        if text:
                            yield text
                    return

        raise GoogleLLMAPIError("Failed to initialize streaming response")

    except Exception as exc:
        raise GoogleLLMAPIError(f"Google Vision LLM streaming request failed: {exc}") from exc


class GoogleVisionLLM:
    """
    Class-based wrapper for Google Gemini Vision LLM.
    
    Example:
        >>> llm = GoogleVisionLLM(model="gemini-3-flash-preview")
        >>> response = llm.invoke(
        ...     prompt="What is in this image?",
        ...     images="path/to/image.jpg"
        ... )
        >>> print(response)
    """
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
        timeout: Optional[float] = 30.0,
        backoff_factor: float = 0.5,
    ):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_factor = backoff_factor
    
    def invoke(self, prompt: str, images: Optional[Union[str, List[str], Any, List[Any]]] = None) -> str:
        """
        Generate a response from the Google Gemini Vision model.
        
        Args:
            prompt: The input prompt text
            images: Image input(s). Can be path(s) or PIL Image(s).
            
        Returns:
            Generated response text.
        """
        result = google_vision_llm(
            prompt=prompt,
            images=images,
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            max_retries=self.max_retries,
            timeout=self.timeout,
            backoff_factor=self.backoff_factor,
        )
        return result["content"]
    
    def stream(self, prompt: str, images: Optional[Union[str, List[str], Any, List[Any]]] = None):
        """
        Generate a streaming response from the Google Gemini Vision model.
        """
        return google_vision_llm_stream(
            prompt=prompt,
            images=images,
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )


__all__ = [
    "google_vision_llm",
    "google_vision_llm_stream",
    "GoogleVisionLLM",
    "GoogleLLMError",
    "GoogleLLMAPIError",
    "GoogleLLMImportError",
    "GoogleLLMResponseError",
]
