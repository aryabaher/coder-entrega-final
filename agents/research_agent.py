"""Investigador: busca en las políticas internas y no calcula."""

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

import modelo
from agents.texto import texto_de
from rag import sistema_rag
from state import AgentState

PROMPT_INVESTIGADOR = (
    "Sos un agente de investigación. Tu única función es buscar datos concretos en "
    "las políticas internas de TechCorp usando tu herramienta. NO hagas cálculos. "
    "Respondé de forma breve con los datos encontrados y su fuente."
)


class ConsultaPoliticas(BaseModel):
    """Entrada de buscar_en_politicas_internas."""

    model_config = ConfigDict(str_strip_whitespace=True)

    pregunta: str = Field(min_length=1, max_length=2000)


class FragmentoPolitica(BaseModel):
    """Un fragmento recuperado, con su fuente."""

    fuente: str = Field(min_length=1)
    contenido: str = Field(min_length=1)


class ResultadoBusqueda(BaseModel):
    """Salida de la herramienta, validada antes de volver al modelo."""

    fragmentos: list[FragmentoPolitica]

    def observacion(self) -> str:
        # La observación del rol tool es siempre un str, no la lista cruda.
        if not self.fragmentos:
            return "No se encontró información relevante en las políticas."
        return "\n\n".join(f"[Fuente: {f.fuente}] {f.contenido}" for f in self.fragmentos)


def _consulta_invalida(error: ValidationError) -> str:
    return (
        "No se encontró información relevante en las políticas. "
        "Pasá la pregunta con texto, por ejemplo 'días de vacaciones por antigüedad'. "
        f"Detalle: {error.errors()[0]['msg']}"
    )


@tool(args_schema=ConsultaPoliticas)
async def buscar_en_politicas_internas(pregunta: str) -> str:
    """Busca información en las políticas internas de TechCorp (vacaciones, teletrabajo,
    seguridad informática, onboarding). Usar para cualquier pregunta sobre reglas o
    números concretos de la empresa."""
    consulta = ConsultaPoliticas(pregunta=pregunta)
    crudos = await (await sistema_rag()).obtener_top_k(consulta.pregunta)
    fragmentos: list[FragmentoPolitica] = []
    for item in crudos:
        try:
            fragmentos.append(FragmentoPolitica(fuente=item["fuente"], contenido=item["contenido"]))
        except ValidationError:
            continue
    return ResultadoBusqueda(fragmentos=fragmentos).observacion()


buscar_en_politicas_internas.handle_validation_error = _consulta_invalida  # type: ignore[attr-defined]


def asegurar_agente() -> Any:
    """create_agent con la herramienta de búsqueda. Se arma una sola vez."""
    global agente_investigador
    if agente_investigador is None:
        agente_investigador = create_agent(
            model=modelo.obtener_llm(),
            tools=[buscar_en_politicas_internas],
            system_prompt=PROMPT_INVESTIGADOR,
        )
    return agente_investigador


agente_investigador: Any = None


async def nodo_investigador(state: AgentState) -> dict[str, Any]:
    """Busca el dato. Recibe solo el último mensaje, no toda la conversación."""
    tarea = state["messages"][-1].content  # contexto acotado, no todo el historial
    agente = asegurar_agente()
    resultado = await modelo.invocar(
        lambda: agente.ainvoke({"messages": [HumanMessage(content=texto_de(tarea))]})
    )
    mensaje = resultado["messages"][-1]
    respuesta = (
        "ERROR: output truncado: investigador cortó la respuesta por max_tokens. Pedile una respuesta más corta."
        if modelo.es_truncada(mensaje)
        else texto_de(mensaje.content)
    )
    return {
        "messages": [AIMessage(content=respuesta, name="investigador")],
        "contribuciones": [{"agente": "investigador", "aporte": respuesta}],
        "pasos": state.get("pasos", 0) + 1,
    }
