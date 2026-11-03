"""Escribe el Mermaid del grafo. No abre Redis ni llama al modelo."""

from requisitos import exigir_python_312

exigir_python_312()

import os
from pathlib import Path

os.environ["TRAZAS_OFFLINE"] = "1"
os.environ.pop("PINECONE_API_KEY", None)


def main() -> None:
    from langgraph.checkpoint.memory import MemorySaver

    from app.graph import compile_graph

    texto = compile_graph(MemorySaver()).get_graph().draw_mermaid()
    destino = Path(__file__).with_name("grafo.mmd")
    destino.write_text(texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()
