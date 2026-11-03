"""Cinco consultas del sistema. Cada una abre una traza `orquestador` en Phoenix.

Si Redis responde, el Checkpointer es AsyncRedisSaver. Si no, esta corrida usa
MemorySaver para poder dejar las trazas igual: en la API el checkpointer es Redis.
"""

from requisitos import exigir_python_312

exigir_python_312()

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from app.observability import init_observability  # noqa: E402

init_observability()

import httpx  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from app.graph import acompile_graph, compile_graph, run_graph, set_graph  # noqa: E402

CONSULTAS = [
    "¿Cuántos días de vacaciones corresponden con 5 años de antigüedad?",
    "¿Cuántos días de teletrabajo por semana permite la política de TechCorp?",
    "¿Qué tiene que completar una persona nueva durante el onboarding?",
    "¿Cuál es la regla de contraseñas de la política de seguridad informática?",
    "Un empleado va a cumplir 12 años en TechCorp. Tuvo 14 días de vacaciones los primeros "
    "5 años, 21 los siguientes 5, y ahora tiene derecho a 28. ¿Cuántos días de vacaciones "
    "acumuló en total en esos 12 años?",
]


async def _limpiar_proyecto() -> None:
    base = os.getenv("PHOENIX_UI", "http://localhost:6006").rstrip("/")
    proyecto = os.getenv("PHOENIX_PROJECT", "intelligence")
    try:
        async with httpx.AsyncClient(timeout=10) as cliente:
            respuesta = await cliente.delete(f"{base}/v1/projects/{proyecto}")
        print(f"Phoenix DELETE {proyecto}: {respuesta.status_code}")
    except httpx.HTTPError as exc:
        print(f"Phoenix DELETE no se pudo: {exc}")


async def _compilar() -> str:
    url = (os.getenv("REDIS_URL") or "redis://localhost:6379").strip()
    try:
        from redis import asyncio as aioredis

        cliente = aioredis.from_url(url, socket_connect_timeout=1)
        try:
            await cliente.ping()
        finally:
            await cliente.aclose()
        set_graph(await acompile_graph())
        return f"AsyncRedisSaver en {url}"
    except Exception as exc:
        set_graph(compile_graph(MemorySaver()))
        return f"MemorySaver ({type(exc).__name__}: Redis no respondió). La API usa AsyncRedisSaver."


async def _trazas_en_phoenix() -> str:
    base = os.getenv("PHOENIX_UI", "http://localhost:6006").rstrip("/")
    proyecto = os.getenv("PHOENIX_PROJECT", "intelligence")
    consulta = """
    query ($id: ID!) {
      node(id: $id) {
        ... on Project {
          name
          traceCount
          spans(first: 20, filterCondition: "name == 'orquestador'") {
            edges { node { name } }
          }
        }
      }
    }
    """
    try:
        async with httpx.AsyncClient(timeout=10) as cliente:
            listado = await cliente.get(f"{base}/v1/projects")
            listado.raise_for_status()
            datos = listado.json().get("data") or []
            elegido = next((item for item in datos if item.get("name") == proyecto), None)
            if elegido is None:
                nombres = ", ".join(str(item.get("name")) for item in datos) or "(ninguno)"
                return f"Phoenix no tiene el proyecto {proyecto}. Proyectos: {nombres}"
            gql = await cliente.post(
                f"{base}/graphql",
                json={"query": consulta, "variables": {"id": elegido["id"]}},
            )
            gql.raise_for_status()
            nodo = (gql.json().get("data") or {}).get("node") or {}
            errores = gql.json().get("errors")
            if errores:
                return f"Phoenix GraphQL: {errores}"
    except httpx.HTTPError as exc:
        return f"Phoenix no respondió en {base}: {exc}"
    orquestador = len((nodo.get("spans") or {}).get("edges") or [])
    return (
        f"Phoenix {base} proyecto {nodo.get('name', proyecto)} "
        f"id={elegido.get('id')} traceCount={nodo.get('traceCount')} spans_orquestador={orquestador}"
    )


async def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    await _limpiar_proyecto()
    print("Checkpointer:", await _compilar())
    from app import worker
    from rag import sistema_rag

    print("RAG:", (await sistema_rag()).origen)

    for indice, pregunta in enumerate(CONSULTAS, start=1):
        hilo = f"prueba-{indice}"
        resultado = await worker._run_traced(
            {"job_id": hilo, "thread_id": hilo},
            pregunta,
            run_graph,
            pregunta,
            {"configurable": {"thread_id": hilo}},
        )
        estado = "PAUSA" if resultado["paused"] else "OK"
        texto = resultado["interrupt"] if resultado["paused"] else resultado["result"]
        print(f"\n[{indice}/5] {estado}")
        print(f"Q: {pregunta}")
        print(f"A: {texto}")
    from opentelemetry import trace

    proveedor = trace.get_tracer_provider()
    flush = getattr(proveedor, "force_flush", None)
    if flush is not None:
        flush(timeout_millis=10000)
    await asyncio.sleep(5)
    print("\n" + await _trazas_en_phoenix())


if __name__ == "__main__":
    asyncio.run(main())
