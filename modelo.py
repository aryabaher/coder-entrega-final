"""Creación del LLM (Google, OpenAI o Anthropic), clasificación de errores y reintentos."""

import asyncio
import inspect
import os
from collections.abc import Awaitable, Callable
from functools import cache
from typing import TypeVar

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError

from app.observability import trace_llm
from errors import ClaveAPIError, ConfiguracionError, CuotaError, OrquestadorError, RedError

MAX_TOKENS = 1024
MAX_INTENTOS = 3
ESPERA_INICIAL_S = 0.5

T = TypeVar("T")

_PROVEEDORES = {
    "google": ("GOOGLE_API_KEY", "GOOGLE_MODEL"),
    "openai": ("OPENAI_API_KEY", "OPENAI_MODEL"),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"),
}


def resolver_modelo(provider: str | None = None, modelo: str | None = None) -> tuple[str, str]:
    """Devuelve (provider, modelo). El nombre del modelo sale del argumento o de la variable de entorno."""
    provider = (provider or os.getenv("LLM_PROVIDER") or "google").strip().lower()
    if provider not in _PROVEEDORES:
        raise ConfiguracionError(
            f"LLM_PROVIDER inválido: '{provider}'. Valores soportados: google, openai, anthropic."
        )
    variable = _PROVEEDORES[provider][1]
    elegido = (modelo or os.getenv(variable) or "").strip()
    if not elegido:
        raise ConfiguracionError(
            f"Falta {variable}. Definila en .env: el nombre del modelo no está clavado en el código."
        )
    return provider, elegido


def crear_llm(
    provider: str | None = None,
    modelo: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
) -> BaseChatModel:
    """Crea el chat model a partir de las variables de entorno.

    Args:
        provider: google, openai o anthropic. Si no se pasa, sale de LLM_PROVIDER (default google).
        modelo: nombre del modelo. Si no se pasa, GOOGLE_MODEL, OPENAI_MODEL o ANTHROPIC_MODEL.
        temperature: 0 para que el ruteo del supervisor sea estable.
        max_tokens: tope de cada respuesta. Una respuesta cortada no se toma como válida.

    Los reintentos del SDK quedan en 0: los hace `invocar`, con backoff y con el mensaje
    de la familia de error. Si falta la clave, el error es 401/key y no se reintenta.
    """
    nombre = (provider or os.getenv("LLM_PROVIDER") or "google").strip().lower()
    if nombre not in _PROVEEDORES:
        raise ConfiguracionError(
            f"LLM_PROVIDER inválido: '{nombre}'. Valores soportados: google, openai, anthropic."
        )
    if nombre == "google" and not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    variable = _PROVEEDORES[nombre][0]
    if not os.getenv(variable):
        raise ClaveAPIError(
            f"401/key: falta {variable} en las variables de entorno. "
            "Copiá .env.example a .env y cargá la clave. No se reintenta."
        )
    provider, modelo = resolver_modelo(nombre, modelo)
    if provider == "google":
        return ChatGoogleGenerativeAI(
            model=modelo, temperature=temperature, max_output_tokens=max_tokens, max_retries=0
        )
    if provider == "anthropic":
        return ChatAnthropic(
            model_name=modelo,
            temperature=temperature,
            max_tokens_to_sample=max_tokens,
            max_retries=0,
            timeout=None,
            stop=None,
        )
    return ChatOpenAI(model=modelo, temperature=temperature, max_completion_tokens=max_tokens, max_retries=0)


@cache
def obtener_llm() -> BaseChatModel:
    """El único LLM del sistema: lo comparten el supervisor y los especialistas.

    Se crea en la primera llamada y no al importar, para que una clave faltante
    salga como el 401/key y no como un error del SDK al cargar.
    """
    return crear_llm()


def es_truncada(mensaje: BaseMessage) -> bool:
    """True si el proveedor cortó la respuesta por max_tokens."""
    metadata = mensaje.response_metadata
    return metadata.get("finish_reason") == "length" or metadata.get("stop_reason") == "max_tokens"


def clasificar_error(exc: BaseException) -> OrquestadorError:
    """Traduce una excepción del SDK (OpenAI, Anthropic, httpx) a la familia de error del sistema."""
    if isinstance(exc, OrquestadorError):
        return exc
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    detalle = f"{type(exc).__name__}: {exc}"
    texto = detalle.lower()
    if status == 401 or "authentication" in texto or "api key" in texto or "api_key" in texto:
        return ClaveAPIError(
            f"401/key: el proveedor rechazó la API key. No se reintenta: revisá la clave en .env. Detalle: {detalle}"
        )
    if status == 429 or "rate limit" in texto or "ratelimit" in texto or "quota" in texto:
        return CuotaError(f"429/cuota: el proveedor limitó las llamadas (rate limit o cuota). Detalle: {detalle}")
    if (
        status in (408, 500, 502, 503, 504)
        or isinstance(exc, (TimeoutError, ConnectionError))
        or "timeout" in texto
        or "timed out" in texto
        or "connection" in texto
    ):
        return RedError(f"red/timeout: no se pudo hablar con el proveedor del LLM. Detalle: {detalle}")
    return OrquestadorError(f"Error del LLM: {detalle}")


@trace_llm
async def invocar(llamada: Callable[[], Awaitable[T] | T], intentos: int = MAX_INTENTOS) -> T:
    """Ejecuta `llamada` y reintenta 429/cuota y red/timeout con backoff (0.5 s, 1 s).

    `llamada` puede devolver un valor o una corrutina (`ainvoke`). Una 401 o un error
    no transitorio se lanzan en el primer intento. Si se agotan los reintentos, el
    mensaje es el del último fallo, no un texto fijo. `GraphRecursionError` pasa sin tocar.
    """
    ultimo: OrquestadorError | None = None
    for intento in range(intentos):
        if intento:
            await asyncio.sleep(ESPERA_INICIAL_S * 2 ** (intento - 1))
        try:
            resultado = llamada()
            if inspect.isawaitable(resultado):
                return await resultado
            return resultado
        except GraphRecursionError:
            raise
        except Exception as exc:
            ultimo = clasificar_error(exc)
            if not ultimo.reintentable:
                if ultimo is exc:
                    raise
                raise ultimo from exc
    assert ultimo is not None
    raise type(ultimo)(f"{ultimo} (reintentos agotados: {intentos} intentos; probá de nuevo en unos segundos)")
