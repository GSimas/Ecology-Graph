import os

try:
    import streamlit as st
except Exception:
    st = None


def _read_streamlit_secret(key):
    if st is None:
        return None
    try:
        value = st.secrets[key]
    except Exception:
        return None
    return value if value not in ("", None) else None


def get_config_value(key, default=None):
    value = os.getenv(key)
    if value not in ("", None):
        return value

    value = _read_streamlit_secret(key)
    if value not in ("", None):
        return value

    return default


def get_required_config(key):
    value = get_config_value(key)
    if value in ("", None):
        raise RuntimeError(
            f"Configuração ausente: defina {key} como variável de ambiente "
            "ou em .streamlit/secrets.toml."
        )
    return value


def get_gemini_api_key(required=False):
    getter = get_required_config if required else get_config_value
    return getter("GEMINI_API_KEY")


def get_neo4j_credentials(required=False):
    getter = get_required_config if required else get_config_value
    uri = getter("NEO4J_URI")
    user = getter("NEO4J_USERNAME")
    password = getter("NEO4J_PASSWORD")
    return uri, user, password
