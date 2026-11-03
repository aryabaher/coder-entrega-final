"""Orquestador async: supervisor, RAG, analista, aprobación y síntesis.

El Checkpointer de LangGraph es AsyncRedisSaver: el hilo sobrevive a un reinicio.
"""

import functools
import os
import unicodedata
from collections.abc import Awaitable, Sequence
from typing import Any, Literal, Protocol

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Overwrite
from pydantic import BaseModel, Field, ValidationError

import modelo
from agents.analyst_agent import nodo_analista
from agents.research_agent import nodo_investigador
from agents.texto import texto_de
from app.hitl import COST, SIDE_EFFECT, approval_node, executor_node, latest_question, route_after_approval
from errors import DecisionInvalidaError, SolicitudInvalidaError
from state import AgentState, Solicitud

MAX_PASOS = 6
RECURSION_LIMIT = 25
TURNOS_DE_CONTEXTO = 3

# USD por millón de tokens. Los defaults son los de gpt-4o-mini; con otro modelo se cambian en .env.
# Phoenix calcula el costo real; esto solo alimenta el umbral de la pausa.
PRECIO_ENTRADA_USD_1M = 0.15
PRECIO_SALIDA_USD_1M = 0.60
COSTO_MAXIMO_USD = 0.01

MARCAS_CRITICAS = (
    "register_payroll_adjustment",
    "ajuste de nomina",
    "registrar ajuste",
    "efecto secundario",
    "costo muy alto",
)

SUPERVISOR_PROMPT = """Sos el Supervisor de un equipo con dos especialistas:
- investigador: busca datos concretos en las políticas internas de TechCorp.
- analista: hace cálculos matemáticos a partir de datos ya obtenidos.

No des por terminada la tarea hasta cumplir las reglas.
Reglas:
1. Si falta un dato de la empresa, mandá a 'investigador'.
2. Si ya se buscó el dato pero falta calcular, mandá a 'analista'.
3. Si ya se buscó Y se calculó lo necesario, respondé 'FINISH'.
4. Nunca repitas el mismo agente dos veces seguidas si ya cumplió su parte.
5. Si la tarea pide registrar un ajuste de nómina u otra acción con efectos secundarios, no la des por terminada: falta la aprobación humana.
{conversacion}
Contribuciones hasta ahora:
{contribuciones}
"""

_contextos: list[Any] = []
_compilado: Any = None


class DecisionSupervisor(BaseModel):
    next: Literal["investigador", "analista", "FINISH"] = Field(
        description="Próximo agente a invocar, o FINISH si la tarea ya está completa"
    )
    razon: str = Field(description="Breve justificación de la decisión")


class Ruta(BaseModel):
    next: Literal["investigador", "analista", "aprobacion", "ejecutor", "FINISH"]
    razon: str


def es_critica(texto: str) -> bool:
    """True si la tarea pide una herramienta con efectos secundarios o anuncia un costo muy alto."""
    normalizado = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()
    return any(marca in normalizado for marca in MARCAS_CRITICAS)


def _numero_del_entorno(variable: str, default: float) -> float:
    """Un número de .env. Si no está o no se puede leer, el default."""
    try:
        return float(os.getenv(variable) or default)
    except ValueError:
        return default


def limite_costo() -> float:
    """COSTO_MAXIMO_USD del entorno."""
    return _numero_del_entorno("COSTO_MAXIMO_USD", COSTO_MAXIMO_USD)


def default_recursion_limit() -> int:
    """RECURSION_LIMIT del entorno. Es el tope de pasos de LangGraph en cada ainvoke."""
    return int(_numero_del_entorno("RECURSION_LIMIT", RECURSION_LIMIT))


def costo_estimado(state: AgentState) -> float:
    entrada = state.get("input_tokens", 0) or 0
    salida = state.get("output_tokens", 0) or 0
    precio_entrada = _numero_del_entorno("PRECIO_ENTRADA_USD_1M", PRECIO_ENTRADA_USD_1M) / 1_000_000
    precio_salida = _numero_del_entorno("PRECIO_SALIDA_USD_1M", PRECIO_SALIDA_USD_1M) / 1_000_000
    return entrada * precio_entrada + salida * precio_salida


class Nodo(Protocol):
    """Un nodo async del grafo. LangGraph le pasa el estado por el nombre `state`."""

    def __call__(self, state: AgentState) -> Awaitable[dict[str, Any]]: ...


def medido(
    nodo: Nodo,
) -> Nodo:
    """Suma al estado los tokens que gastó el nodo. Con eso decide la guarda de costo."""

    @functools.wraps(nodo)
    async def envoltura(state: AgentState) -> dict[str, Any]:
        with get_usage_metadata_callback() as uso:
            salida = await nodo(state)
        entrada = sum(int(u.get("input_tokens", 0)) for u in uso.usage_metadata.values())
        generados = sum(int(u.get("output_tokens", 0)) for u in uso.usage_metadata.values())
        return {**salida, "input_tokens": entrada, "output_tokens": generados}

    return envoltura


def con_contexto(
    nodo: Nodo,
) -> Nodo:
    """En un turno de seguimiento, el especialista recibe la pregunta con la conversación previa.

    El especialista solo lee el último mensaje. Sin esto, "¿y si fueran 15 años?" llega sin el tema.
    """

    @functools.wraps(nodo)
    async def envoltura(state: AgentState) -> dict[str, Any]:
        previa = conversacion_previa(state["messages"])
        ultimo = state["messages"][-1]
        if not previa or ultimo.type != "human":
            return await nodo(state)
        pregunta = HumanMessage(content=f"{previa}\nPregunta actual: {texto_de(ultimo.content)}")
        return await nodo({**state, "messages": [*state["messages"][:-1], pregunta]})

    return envoltura


def con_tarea(
    nodo: Nodo,
) -> Nodo:
    """El analista recibe la pregunta del turno y los datos que ya consiguió el equipo.

    Con solo el último mensaje, a veces calcula sin los años de la pregunta o sin los datos de la política.
    """

    @functools.wraps(nodo)
    async def envoltura(state: AgentState) -> dict[str, Any]:
        datos = "\n".join(
            f"- {c['agente']}: {c['aporte']}"
            for c in state.get("contribuciones", [])
            if not c["aporte"].startswith(("ERROR", "No se encontró"))
        )
        tarea = (
            f"{conversacion_previa(state['messages'])}"
            f"Pregunta actual: {latest_question(state['messages'])}\n"
            f"Datos del equipo:\n{datos or '(todavía ninguno: usá los números de la pregunta)'}"
        )
        return await nodo({**state, "messages": [*state["messages"], HumanMessage(content=tarea)]})

    return envoltura


def conversacion_previa(mensajes: Sequence[BaseMessage]) -> str:
    """Preguntas y respuestas finales de los turnos anteriores del hilo (las últimas TURNOS_DE_CONTEXTO)."""
    pares: list[tuple[str, str]] = []
    pregunta: str | None = None
    for mensaje in mensajes:
        if mensaje.type == "human":
            pregunta = texto_de(mensaje.content)
        elif mensaje.type == "ai" and getattr(mensaje, "name", None) == "supervisor" and pregunta is not None:
            pares.append((pregunta, texto_de(mensaje.content)))
            pregunta = None
    if not pares:
        return ""
    lineas = [f"- Pregunta: {p}\n  Respuesta: {r}" for p, r in pares[-TURNOS_DE_CONTEXTO:]]
    return "Conversación previa del hilo:\n" + "\n".join(lineas) + "\n"


async def nodo_supervisor(state: AgentState) -> dict[str, Any]:
    """Router: quién interviene ahora, o FINISH si la tarea ya está completa."""
    if state.get("pasos", 0) >= MAX_PASOS:
        return {"next_agent": "FINISH", "task_completed": True}

    costo = costo_estimado(state)
    limite = limite_costo()
    if costo > limite and state.get("cost_approved") is None:
        return {
            "next_agent": "aprobacion",
            "approval_reason": COST,
            "estimated_cost_usd": round(costo, 6),
            "cost_limit_usd": limite,
            "task_completed": False,
        }

    contribuciones_texto = "\n".join(
        f"- {c['agente']}: {c['aporte'][:200]}" for c in state.get("contribuciones", [])
    ) or "(ninguna todavía)"
    pregunta_original = latest_question(state["messages"])
    previa = conversacion_previa(state["messages"])
    prompt_sistema = SUPERVISOR_PROMPT.format(
        contribuciones=contribuciones_texto,
        conversacion=f"\n{previa}" if previa else "",
    )
    llm_supervisor = modelo.obtener_llm().with_structured_output(DecisionSupervisor)
    decision = await modelo.invocar(
        lambda: llm_supervisor.ainvoke(
            [
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": f"Tarea original: {pregunta_original}"},
            ]
        )
    )
    if not isinstance(decision, DecisionSupervisor):
        raise DecisionInvalidaError(
            "Decisión inválida del supervisor: "
            f"{decision!r}. No se adivina una ruta: revisá el modelo configurado en LLM_PROVIDER."
        )
    ruta = aplicar_guardas(state, decision, pregunta_original)
    salida: dict[str, Any] = {"next_agent": ruta.next, "task_completed": ruta.next == "FINISH"}
    if ruta.next == "aprobacion":
        salida["approval_reason"] = SIDE_EFFECT
    return salida


def _cumplio(state: AgentState, agente: str) -> bool:
    """True si ese agente ya dejó un aporte usable. Un error o un 'no encontrado' no cuenta."""
    return any(
        c["agente"] == agente and not c["aporte"].startswith(("ERROR", "No se encontró"))
        for c in state.get("contribuciones", [])
    )


def _guardas_equipo(state: AgentState, decision: DecisionSupervisor, pregunta: str) -> DecisionSupervisor:
    """FINISH solo después de buscar y, si la pregunta pide una cuenta, de calcular."""
    hechos = {c["agente"] for c in state.get("contribuciones", [])}
    if decision.next == "investigador" and _cumplio(state, "investigador"):
        if "analista" not in hechos and _pide_calculo(pregunta):
            return DecisionSupervisor(
                next="analista", razon="[guarda] el investigador ya cumplió su parte: sigue el analista."
            )
        return DecisionSupervisor(next="FINISH", razon="[guarda] el investigador ya cumplió su parte.")
    if decision.next == "analista" and _cumplio(state, "analista"):
        return DecisionSupervisor(next="FINISH", razon="[guarda] el analista ya cumplió su parte.")
    if decision.next == "analista" and not _pide_calculo(pregunta):
        if _cumplio(state, "investigador"):
            return DecisionSupervisor(next="FINISH", razon="[guarda] no hay una cuenta para hacer: alcanza con el dato.")
        return DecisionSupervisor(next="investigador", razon="[guarda] no hay una cuenta para hacer: falta el dato.")
    if decision.next != "FINISH":
        return decision
    if "investigador" not in hechos:
        return DecisionSupervisor(
            next="investigador", razon="[guarda] FINISH sin datos de las políticas: primero el investigador."
        )
    if "analista" not in hechos and _pide_calculo(pregunta):
        return DecisionSupervisor(next="analista", razon="[guarda] FINISH sin el cálculo: primero el analista.")
    return decision


def _lista_para_efecto(state: AgentState, pregunta: str) -> bool:
    if not _cumplio(state, "investigador"):
        return False
    if _pide_calculo(pregunta) and not _cumplio(state, "analista"):
        return False
    return True


def _guarda_critica(state: AgentState, decision: DecisionSupervisor, pregunta: str) -> Ruta:
    """Antes de la herramienta con efectos secundarios, la ruta es el nodo de aprobación."""
    if not state.get("critical"):
        return Ruta(next=decision.next, razon=decision.razon)
    if state.get("approved") is False or state.get("side_effect_done"):
        return Ruta(next="FINISH", razon="[guarda] la tarea crítica ya se resolvió.")
    if state.get("approved") is True:
        return Ruta(next="ejecutor", razon="[guarda] la persona aprobó la herramienta con efectos secundarios.")
    if _lista_para_efecto(state, pregunta):
        return Ruta(next="aprobacion", razon="[guarda] herramienta con efectos secundarios: hace falta aprobación.")
    return Ruta(next=decision.next, razon=decision.razon)


def aplicar_guardas(state: AgentState, decision: DecisionSupervisor, pregunta: str) -> Ruta:
    """Las guardas del equipo y, si la tarea es crítica, la pausa antes del efecto."""
    return _guarda_critica(state, _guardas_equipo(state, decision, pregunta), pregunta)


PALABRAS_DE_CALCULO = ("total", "acumul", "calcul", "suma", "promedio", "diferencia", "multiplic")


def _pide_calculo(pregunta: str) -> bool:
    """True si la pregunta pide una cuenta. "Cuántos" o un número suelto (un legajo, una antigüedad) no alcanzan."""
    texto = pregunta.lower()
    return any(palabra in texto for palabra in PALABRAS_DE_CALCULO)


def enrutar(state: AgentState) -> Literal["investigador", "analista", "aprobacion", "ejecutor", "sintesis"]:
    if state["next_agent"] == "FINISH":
        return "sintesis"
    destino = state["next_agent"]
    if destino not in ("investigador", "analista", "aprobacion", "ejecutor"):
        raise DecisionInvalidaError(
            f"Decisión inválida del supervisor: {destino!r}. "
            "Tiene que ser investigador, analista, aprobacion, ejecutor o FINISH."
        )
    return destino  # type: ignore[return-value]


async def nodo_sintesis(state: AgentState) -> dict[str, Any]:
    """Redacta la respuesta final con la pregunta, el hilo y las contribuciones. No busca ni calcula."""
    aportes = [f"- [{c['agente']}] {c['aporte']}" for c in state.get("contribuciones", [])]
    if state.get("side_effect_result"):
        aportes.append(
            "- [ejecutor] La persona aprobó y se registró el ajuste de nómina. "
            f"Salida de la herramienta: {state['side_effect_result']}"
        )
    if state.get("approved") is False:
        aportes.append("- [aprobacion] La persona rechazó el ajuste de nómina: no se registró.")
    if state.get("cost_approved") is False:
        aportes.append("- [aprobacion] La persona cortó la ejecución por costo: se responde con lo obtenido.")
    hubo_decision = bool(state.get("side_effect_result")) or False in (state.get("approved"), state.get("cost_approved"))
    previa = conversacion_previa(state["messages"])
    prompt = (
        f"{previa}\nPregunta actual: {latest_question(state['messages'])}\n\nAportes del equipo:\n"
        + "\n".join(aportes)
        + "\n\nRedactá una respuesta final clara y breve combinando estos aportes. Respondé solo lo que se pregunta."
        + " Escribí texto plano, sin LaTeX ni Markdown. No agregues cuentas que no estén en los aportes."
        + (" Decí qué decidió la persona sobre la acción pausada." if hubo_decision else "")
    )
    llm = modelo.obtener_llm()
    respuesta_final = await modelo.invocar(lambda: llm.ainvoke(prompt))
    return {"messages": [AIMessage(content=texto_de(respuesta_final.content), name="supervisor")]}


grafo = StateGraph(AgentState)
grafo.add_node("supervisor", medido(nodo_supervisor))
grafo.add_node("investigador", medido(con_contexto(nodo_investigador)))
grafo.add_node("analista", medido(con_tarea(nodo_analista)))
grafo.add_node("aprobacion", approval_node)
grafo.add_node("ejecutor", executor_node)
grafo.add_node("sintesis", medido(nodo_sintesis))

grafo.add_edge(START, "supervisor")
grafo.add_conditional_edges(
    "supervisor",
    enrutar,
    {
        "investigador": "investigador",
        "analista": "analista",
        "aprobacion": "aprobacion",
        "ejecutor": "ejecutor",
        "sintesis": "sintesis",
    },
)
grafo.add_edge("investigador", "supervisor")
grafo.add_edge("analista", "supervisor")
# La arista que sale de aprobacion solo llega al efecto si la persona aprobó.
grafo.add_conditional_edges(
    "aprobacion",
    route_after_approval,
    {"ejecutor": "ejecutor", "supervisor": "supervisor", "sintesis": "sintesis"},
)
grafo.add_edge("ejecutor", "supervisor")
grafo.add_edge("sintesis", END)

type Orquestador = CompiledStateGraph[AgentState, None, AgentState, AgentState]


async def abrir_redis_saver(url: str | None = None) -> AsyncRedisSaver:
    """Checkpointer de LangGraph. __aenter__ llama a asetup() y crea los índices en Redis."""
    destino = (url or os.getenv("REDIS_URL") or "redis://localhost:6379").strip()
    contexto = AsyncRedisSaver.from_conn_string(destino)
    saver = await contexto.__aenter__()
    _contextos.append(contexto)
    return saver


def compile_graph(checkpointer: BaseCheckpointSaver[Any]) -> Orquestador:
    """Una sola compilación. El checkpointer lo abre acompile_graph (AsyncRedisSaver) o el test (MemorySaver)."""
    return grafo.compile(checkpointer=checkpointer)


async def acompile_graph(checkpointer: BaseCheckpointSaver[Any] | None = None) -> Orquestador:
    """Compilación de producción. Sin argumento, el Checkpointer es AsyncRedisSaver."""
    if checkpointer is None:
        checkpointer = await abrir_redis_saver()
    return compile_graph(checkpointer)


def set_graph(compilado: Any) -> None:
    global _compilado
    _compilado = compilado


def graph_ready() -> bool:
    return _compilado is not None


def get_graph() -> Any:
    if _compilado is None:
        raise RuntimeError("El grafo no está compilado. Llamá a acompile_graph() con AsyncRedisSaver o a set_graph().")
    return _compilado


def validar_pregunta(texto: str) -> str:
    """Pydantic antes del grafo: una pregunta vacía no gasta llamadas."""
    try:
        return Solicitud(texto=texto).texto
    except ValidationError as exc:
        campos = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise SolicitudInvalidaError(f"Solicitud inválida: {campos}") from exc


def estado_inicial(pregunta: str) -> dict[str, Any]:
    """Entrada de un turno. messages se acumula en el hilo; lo del turno se reinicia con Overwrite."""
    texto = validar_pregunta(pregunta)
    return {
        "messages": [HumanMessage(content=texto)],
        "next_agent": None,
        "contribuciones": Overwrite([]),
        "pasos": 0,
        "task_completed": False,
        "critical": es_critica(texto),
        "approval_reason": None,
        "approved": None,
        "cost_approved": None,
        "side_effect_done": False,
        "side_effect_result": None,
        "input_tokens": Overwrite(0),
        "output_tokens": Overwrite(0),
        "estimated_cost_usd": None,
        "cost_limit_usd": None,
    }


async def leer_interrupcion(config: dict[str, Any]) -> dict[str, Any] | None:
    """Payload de interrupt, si el grafo quedó pausado en el nodo de aprobación."""
    snap = await get_graph().aget_state(config)
    crudos: list[Any] = list(getattr(snap, "interrupts", None) or ())
    if not crudos:
        for tarea in getattr(snap, "tasks", ()) or ():
            crudos.extend(getattr(tarea, "interrupts", ()) or ())
    if not crudos:
        return None
    valor = getattr(crudos[0], "value", crudos[0])
    if isinstance(valor, dict):
        return valor
    return {"reason": str(valor)}


async def _empaquetar(resultado: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    pausa = await leer_interrupcion(config)
    if pausa is not None:
        return {"paused": True, "interrupt": pausa, "result": None}
    mensajes = resultado.get("messages") or []
    texto = texto_de(mensajes[-1].content) if mensajes else ""
    return {"paused": False, "interrupt": None, "result": texto}


async def run_graph(pregunta: str, config: dict[str, Any], recursion_limit: int | None = None) -> dict[str, Any]:
    """Un turno del hilo config['configurable']['thread_id']. recursion_limit llega al ainvoke.

    Sin argumento, sale de RECURSION_LIMIT en .env (default 25).
    """
    conf = {**config, "recursion_limit": recursion_limit or default_recursion_limit()}
    resultado = await get_graph().ainvoke(estado_inicial(pregunta), conf)
    return await _empaquetar(resultado, conf)


async def resume_graph(
    decision: dict[str, Any], config: dict[str, Any], recursion_limit: int | None = None
) -> dict[str, Any]:
    """Sigue después de interrupt. El dict es lo que interrupt() devuelve en el nodo."""
    conf = {**config, "recursion_limit": recursion_limit or default_recursion_limit()}
    resultado = await get_graph().ainvoke(Command(resume=decision), conf)
    return await _empaquetar(resultado, conf)


async def thread_history(thread_id: str) -> list[dict[str, str | None]] | None:
    """Mensajes del hilo, leídos del Checkpointer. None si el hilo no tiene checkpoints."""
    snap = await get_graph().aget_state({"configurable": {"thread_id": thread_id}})
    mensajes = (getattr(snap, "values", None) or {}).get("messages") or []
    if not mensajes:
        return None
    return [
        {"role": m.type, "name": getattr(m, "name", None), "content": texto_de(m.content)}
        for m in mensajes
    ]
