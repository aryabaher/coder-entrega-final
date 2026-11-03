"""Analista: calcula con la calculadora segura y no busca información nueva."""

import ast
import operator as op
from collections.abc import Callable
from typing import Any, cast

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

import modelo
from agents.texto import texto_de
from state import AgentState

PROMPT_ANALISTA = (
    "Sos un agente de análisis matemático. Tu única función es calcular usando tu "
    "herramienta a partir de los números que te den. NO busques información nueva. "
    "Si hay tramos (cantidad de años con cierta cantidad de días por año), armá la expresión "
    "como suma de productos, años por días de cada tramo, por ejemplo '5*14 + 5*21 + 2*28', "
    "y verificá antes que los años de los tramos sumen el total de la pregunta. "
    "Respondé con el resultado y la expresión que usaste."
)

_OPERADORES = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}
MAX_EXPONENTE = 100


class ExpresionCalculo(BaseModel):
    """Entrada de la calculadora."""

    model_config = ConfigDict(str_strip_whitespace=True)

    expresion: str = Field(min_length=1, max_length=200)


class ResultadoCalculo(BaseModel):
    """Salida de la calculadora. `error` trae cómo seguir; si no, `valor` es el número."""

    expresion: str
    valor: str = ""
    error: str | None = None

    def observacion(self) -> str:
        # La observación del rol tool es siempre un str.
        return self.error or f"Resultado: {self.valor}"


def _expresion_invalida(error: ValidationError) -> str:
    return (
        "Error al calcular '': la expresión está vacía. "
        "Pasá solo números y + - * / **, por ejemplo '5*14 + 5*21 + 2*28'."
        if "at least 1 character" in error.errors()[0]["msg"]
        else f"Error al calcular: {error.errors()[0]['msg']}. Corregí la expresión y volvé a llamar a calculadora."
    )


def _evaluar_nodo(nodo: ast.expr) -> int | float:
    if isinstance(nodo, ast.Constant) and isinstance(nodo.value, (int, float)) and not isinstance(nodo.value, bool):
        return nodo.value
    if isinstance(nodo, ast.BinOp) and type(nodo.op) in _OPERADORES:
        derecha = nodo.right
        if isinstance(nodo.op, ast.Pow) and isinstance(derecha, ast.Constant) and isinstance(derecha.value, (int, float)) and abs(derecha.value) > MAX_EXPONENTE:
            raise ValueError(f"exponente fuera de rango (máximo {MAX_EXPONENTE})")
        cuenta = cast(Callable[[int | float, int | float], int | float], _OPERADORES[type(nodo.op)])
        return cuenta(_evaluar_nodo(nodo.left), _evaluar_nodo(nodo.right))
    if isinstance(nodo, ast.UnaryOp) and type(nodo.op) in _OPERADORES:
        signo = cast(Callable[[int | float], int | float], _OPERADORES[type(nodo.op)])
        return signo(_evaluar_nodo(nodo.operand))
    raise ValueError("Expresión no permitida")


@tool(args_schema=ExpresionCalculo)
def calculadora(expresion: str) -> str:
    """Evalúa una expresión matemática (+, -, *, /, **, paréntesis). Ejemplo: '5*14 + 5*21 + 2*28'."""
    pedido = ExpresionCalculo(expresion=expresion)
    texto = pedido.expresion.replace("$", "").strip()
    if "," in texto or _tiene_miles(texto):
        return ResultadoCalculo(
            expresion=pedido.expresion,
            error=(
                f"Error al calcular '{pedido.expresion}': el número tiene separador de miles. "
                "Escribilo sin puntos ni comas, por ejemplo 25000, y volvé a llamar a calculadora."
            ),
        ).observacion()
    try:
        resultado = _evaluar_nodo(ast.parse(texto, mode="eval").body)
    except ZeroDivisionError:
        return ResultadoCalculo(
            expresion=pedido.expresion,
            error=(
                f"Error al calcular '{pedido.expresion}': división por cero. "
                "Corregí la expresión y volvé a llamar a calculadora."
            ),
        ).observacion()
    except (ValueError, SyntaxError, KeyError) as exc:
        detalle = exc.msg if isinstance(exc, SyntaxError) else exc
        return ResultadoCalculo(
            expresion=pedido.expresion,
            error=(
                f"Error al calcular '{pedido.expresion}': {detalle}. "
                "Solo números y + - * / ** con paréntesis. Corregí la expresión y volvé a llamar a calculadora."
            ),
        ).observacion()
    if isinstance(resultado, float) and resultado.is_integer():
        resultado = int(resultado)
    return ResultadoCalculo(expresion=pedido.expresion, valor=str(resultado)).observacion()


calculadora.handle_validation_error = _expresion_invalida  # type: ignore[assignment]


def _tiene_miles(texto: str) -> bool:
    import re

    return re.search(r"(?<![\d.])[1-9]\d{0,2}(?:\.\d{3})+(?![\d.])", texto) is not None


def asegurar_agente() -> Any:
    """create_agent con la calculadora. Se arma una sola vez."""
    global agente_analista
    if agente_analista is None:
        agente_analista = create_agent(
            model=modelo.obtener_llm(),
            tools=[calculadora],
            system_prompt=PROMPT_ANALISTA,
        )
    return agente_analista


agente_analista: Any = None


async def nodo_analista(state: AgentState) -> dict[str, Any]:
    """Calcula. Recibe solo el último mensaje, no toda la conversación."""
    tarea = state["messages"][-1].content  # contexto acotado, no todo el historial
    agente = asegurar_agente()
    resultado = await modelo.invocar(
        lambda: agente.ainvoke({"messages": [HumanMessage(content=texto_de(tarea))]})
    )
    mensaje = resultado["messages"][-1]
    respuesta = (
        "ERROR: output truncado: analista cortó la respuesta por max_tokens. Pedile una respuesta más corta."
        if modelo.es_truncada(mensaje)
        else texto_de(mensaje.content)
    )
    return {
        "messages": [AIMessage(content=respuesta, name="analista")],
        "contribuciones": [{"agente": "analista", "aporte": respuesta}],
        "pasos": state.get("pasos", 0) + 1,
    }
