from functools import lru_cache

from app_config import get_gemini_api_key

DEFAULT_TEXT_MODELS = ("gemini-2.5-flash",)
DEFAULT_FAST_MODELS = ("gemini-2.5-flash-lite", "gemini-2.5-flash",)

USE_NEW_SDK = False

try:
    from google import genai
    from google.genai import types
    USE_NEW_SDK = True
except ImportError:
    import google.generativeai as legacy_genai


@lru_cache(maxsize=8)
def get_genai_client(api_key):
    if USE_NEW_SDK:
        return genai.Client(api_key=api_key)

    legacy_genai.configure(api_key=api_key)
    return legacy_genai


def _build_config(
    temperature=0.3,
    system_instruction=None,
    response_mime_type=None,
    candidate_count=None,
):
    if not USE_NEW_SDK:
        return legacy_genai.types.GenerationConfig(
            temperature=temperature,
            candidate_count=candidate_count,
            response_mime_type=response_mime_type,
        )

    config_kwargs = {"temperature": temperature}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if response_mime_type:
        config_kwargs["response_mime_type"] = response_mime_type
    if candidate_count is not None:
        config_kwargs["candidate_count"] = candidate_count
    return types.GenerateContentConfig(**config_kwargs)


def generate_content(
    *,
    prompt=None,
    contents=None,
    api_key=None,
    model_candidates=None,
    temperature=0.3,
    system_instruction=None,
    response_mime_type=None,
    candidate_count=None,
    stream=False,
):
    resolved_api_key = api_key or get_gemini_api_key(required=True)
    payload = contents if contents is not None else prompt
    if payload is None:
        raise ValueError("Forneça `prompt` ou `contents` para o Gemini.")

    client = get_genai_client(resolved_api_key)
    config = _build_config(
        temperature=temperature,
        system_instruction=system_instruction,
        response_mime_type=response_mime_type,
        candidate_count=candidate_count,
    )

    last_error = None
    for model_name in model_candidates or DEFAULT_TEXT_MODELS:
        try:
            if USE_NEW_SDK:
                if stream:
                    return client.models.generate_content_stream(
                        model=model_name,
                        contents=payload,
                        config=config,
                    )
                return client.models.generate_content(
                    model=model_name,
                    contents=payload,
                    config=config,
                )

            model = client.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction,
                generation_config=config,
            )
            return model.generate_content(payload, stream=stream)
        except Exception as exc:
            last_error = exc

    raise last_error if last_error else RuntimeError("Falha ao chamar a API Gemini.")


def response_text(response):
    text = getattr(response, "text", None)
    if text:
        return text
    return ""


def content_from_text(role, text):
    if USE_NEW_SDK:
        return types.Content(
            role=role,
            parts=[types.Part.from_text(text=text)],
        )

    return {"role": role, "parts": [text]}
