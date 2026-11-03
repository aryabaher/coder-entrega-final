"""Lo configurable sale del entorno y llega a la librería; los contratos de la API rechazan estados inventados."""

import asyncio
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from langchain_core.messages import HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import ValidationError

import rag
from app import graph, hitl
from errors import ConfiguracionError

RAIZ = Path(__file__).resolve().parents[1]


def test_dimension_y_ubicacion_de_pinecone_salen_del_entorno(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    monkeypatch.delenv("PINECONE_CLOUD", raising=False)
    monkeypatch.delenv("PINECONE_REGION", raising=False)
    assert rag.dimension_embedding() == 1536
    assert rag.ubicacion_pinecone() == ("aws", "us-east-1")
    monkeypatch.setenv("EMBEDDING_DIM", "3072")
    monkeypatch.setenv("PINECONE_CLOUD", "gcp")
    monkeypatch.setenv("PINECONE_REGION", "europe-west4")
    assert rag.dimension_embedding() == 3072
    assert rag.ubicacion_pinecone() == ("gcp", "europe-west4")
    monkeypatch.setenv("EMBEDDING_DIM", "ancho")
    with pytest.raises(ConfiguracionError, match="EMBEDDING_DIM inválido"):
        rag.dimension_embedding()


def test_conectar_pinecone_pasa_el_entorno_a_la_libreria(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin red: Pinecone, los embeddings y el vector store son dobles que anotan lo que reciben."""
    import langchain_openai
    import langchain_pinecone
    import pinecone

    visto: dict[str, Any] = {}

    class EmbeddingsDoble(rag.EmbeddingsLocales):
        def __init__(self, model: str, dimensions: int) -> None:
            super().__init__(dimension=dimensions)
            visto["embeddings"] = (model, dimensions)

    class IndiceDoble:
        def describe_index_stats(self) -> dict[str, Any]:
            return {"namespaces": {}}

    class PineconeDoble:
        def __init__(self, api_key: str) -> None:
            pass

        def list_indexes(self) -> list[dict[str, str]]:
            return []

        def create_index(self, **kwargs: Any) -> None:
            visto["create_index"] = kwargs

        def Index(self, nombre: str) -> IndiceDoble:
            return IndiceDoble()

    class VectorStoreDoble:
        def __init__(self, index_name: str, embedding: Any, namespace: str) -> None:
            visto["vectorstore"] = (index_name, namespace)
            self.memoria = InMemoryVectorStore(embedding)

        def add_documents(self, docs: list[Any]) -> None:
            visto["ingestados"] = len(docs)

        def as_retriever(self, search_kwargs: dict[str, Any]) -> Any:
            visto["search_kwargs"] = search_kwargs
            return self.memoria.as_retriever(search_kwargs={"k": search_kwargs["k"]})

    monkeypatch.setattr(langchain_openai, "OpenAIEmbeddings", EmbeddingsDoble)
    monkeypatch.setattr(langchain_pinecone, "PineconeVectorStore", VectorStoreDoble)
    monkeypatch.setattr(pinecone, "Pinecone", PineconeDoble)
    for variable, valor in {
        "PINECONE_API_KEY": "pinecone-de-test",
        "OPENAI_API_KEY": "clave-de-test",
        "INDEX_NAME": "indice-de-test",
        "PINECONE_NAMESPACE": "ns-test",
        "OPENAI_EMBEDDING_MODEL": "text-embedding-3-large",
        "EMBEDDING_DIM": "3072",
        "PINECONE_CLOUD": "gcp",
        "PINECONE_REGION": "europe-west4",
    }.items():
        monkeypatch.setenv(variable, valor)

    sistema = rag.conectar_pinecone(k=3)

    spec = visto["create_index"]["spec"]
    assert visto["create_index"]["name"] == "indice-de-test"
    assert visto["create_index"]["dimension"] == 3072
    assert visto["create_index"]["metric"] == "cosine"
    assert (spec.cloud, spec.region) == ("gcp", "europe-west4")
    assert visto["embeddings"] == ("text-embedding-3-large", 3072)
    assert visto["vectorstore"] == ("indice-de-test", "ns-test")
    assert visto["search_kwargs"] == {"k": 3, "namespace": "ns-test"}
    assert visto["ingestados"] > 0
    assert sistema.k == 3


def test_falta_el_modelo_dice_que_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    import modelo

    monkeypatch.setenv("OPENAI_API_KEY", "clave-de-test")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(ConfiguracionError, match="^Falta OPENAI_MODEL. Definila en .env"):
        modelo.crear_llm("openai")


def test_indice_incompatible_no_se_reingesta(monkeypatch: pytest.MonkeyPatch) -> None:
    from errors import IndiceIncompatibleError

    monkeypatch.setenv("INDEX_NAME", "indice-de-test")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    rag.verificar_indice({"dimension": 1536, "metric": "cosine"}, 1536)
    with pytest.raises(IndiceIncompatibleError, match="es 3072D y text-embedding-3-small usa 1536D. No se reingesta"):
        rag.verificar_indice({"dimension": 3072, "metric": "cosine"}, 1536)
    with pytest.raises(IndiceIncompatibleError, match="Mismatch de métrica"):
        rag.verificar_indice({"dimension": 1536, "metric": "euclidean"}, 1536)


def test_el_precio_por_token_sale_del_entorno(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRECIO_ENTRADA_USD_1M", "2.5")
    monkeypatch.setenv("PRECIO_SALIDA_USD_1M", "10")
    costo = graph.costo_estimado({"input_tokens": 1_000_000, "output_tokens": 1_000_000})  # type: ignore[typeddict-item]
    assert costo == pytest.approx(12.5)


def test_recursion_limit_del_entorno_llega_al_invoke(monkeypatch: pytest.MonkeyPatch) -> None:
    visto: dict[str, Any] = {}

    class DobleGrafo:
        async def ainvoke(self, entrada: object, config: dict[str, Any]) -> dict[str, Any]:
            visto["config"] = config
            return {"messages": []}

        async def aget_state(self, config: dict[str, Any]) -> object:
            class Snap:
                interrupts: tuple[()] = ()
                tasks: tuple[()] = ()
                values: dict[str, Any] = {"messages": []}

            return Snap()

    monkeypatch.setenv("RECURSION_LIMIT", "40")
    graph.set_graph(DobleGrafo())
    asyncio.run(graph.run_graph("hola", {"configurable": {"thread_id": "limite-env"}}))
    assert graph.default_recursion_limit() == 40
    assert visto["config"]["recursion_limit"] == 40


def test_el_ejecutor_pasa_por_la_herramienta(monkeypatch: pytest.MonkeyPatch) -> None:
    entradas: list[dict[str, str]] = []

    class HerramientaEspia:
        async def ainvoke(self, entrada: dict[str, str]) -> str:
            entradas.append(entrada)
            return "Adjustment registered: espia"

    monkeypatch.setattr(hitl, "register_payroll_adjustment", HerramientaEspia())
    estado = {"approved": True, "messages": [HumanMessage(content="Registrá el ajuste de nómina del legajo 1042")]}
    salida = asyncio.run(hitl.executor_node(estado))  # type: ignore[arg-type]
    assert entradas == [{"detail": "Registrá el ajuste de nómina del legajo 1042"}]
    assert salida == {"side_effect_done": True, "side_effect_result": "Adjustment registered: espia"}


async def test_la_herramienta_con_efecto_valida_con_su_schema() -> None:
    assert await hitl.register_payroll_adjustment.ainvoke({"detail": "   "}) == (
        "Adjustment not registered: the detail is empty."
    )
    assert hitl.side_effects == []


def test_un_status_inventado_no_pasa_la_validacion() -> None:
    from app.main import TaskResponse, TaskStatus

    registro = {"job_id": "j", "thread_id": "t", "status": "DONE", "query": "q"}
    assert TaskStatus.model_validate(registro).status == "DONE"
    with pytest.raises(ValidationError):
        TaskStatus.model_validate({**registro, "status": "LOST"})
    with pytest.raises(ValidationError):
        TaskResponse(job_id="j", status="QUEUED", thread_id="t")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        hitl.ApprovalResponse(job_id="j", status="OK")  # type: ignore[arg-type]


def test_sin_requests_y_pinecone_fuera_del_loop() -> None:
    fuentes = [p for p in RAIZ.glob("**/*.py") if ".venv" not in p.parts and "tests" not in p.parts]
    con_requests = [
        p.name for p in fuentes if re.search(r"^\s*(import|from)\s+requests\b", p.read_text(encoding="utf-8"), re.M)
    ]
    assert con_requests == []
    assert "await asyncio.to_thread(construir_rag)" in (RAIZ / "rag.py").read_text(encoding="utf-8")


def test_el_compose_persiste_redis_y_phoenix() -> None:
    compose = yaml.safe_load((RAIZ / "docker-compose.yml").read_text(encoding="utf-8"))
    servicios = compose["services"]
    assert "redis-data:/data" in servicios["redis"]["volumes"]
    assert servicios["redis"]["command"] == ["redis-server", "--appendonly", "yes"]
    assert "phoenix-data:/mnt/data" in servicios["phoenix"]["volumes"]
    pin = next(
        linea.split("==")[1]
        for linea in (RAIZ / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if linea.startswith("arize-phoenix==")
    )
    assert servicios["phoenix"]["image"] == f"arizephoenix/phoenix:version-{pin}"
    assert {"redis-data", "phoenix-data"} <= set(compose["volumes"])
