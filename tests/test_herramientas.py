"""Pydantic en las herramientas y k que llega al retriever."""

import pytest
from pydantic import ValidationError

from agents.analyst_agent import calculadora
from agents.research_agent import buscar_en_politicas_internas
from rag import construir_rag_local


async def test_la_busqueda_valida_la_entrada_y_devuelve_fuente() -> None:
    texto = await buscar_en_politicas_internas.ainvoke({"pregunta": "días de vacaciones por antigüedad"})
    assert "[Fuente:" in texto
    assert "politica_vacaciones.txt" in texto


async def test_la_pregunta_vacia_dice_como_seguir() -> None:
    texto = await buscar_en_politicas_internas.ainvoke({"pregunta": "   "})
    assert "Pasá la pregunta con texto" in texto


def test_la_calculadora_valida_y_resuelve_los_12_anos() -> None:
    assert calculadora.invoke({"expresion": "5*14 + 5*21 + 2*28"}) == "Resultado: 231"


def test_la_calculadora_vacia_dice_como_seguir() -> None:
    texto = calculadora.invoke({"expresion": "  "})
    assert "la expresión está vacía" in texto


def test_la_calculadora_rechaza_la_division_por_cero() -> None:
    texto = calculadora.invoke({"expresion": "1/0"})
    assert "división por cero" in texto
    assert "volvé a llamar a calculadora" in texto


def test_el_ajuste_vacio_no_se_registra() -> None:
    from app.hitl import AdjustmentResult, PayrollAdjustment, _register, side_effects

    antes = len(side_effects)
    with pytest.raises(ValidationError):
        PayrollAdjustment(detail="  ")
    assert _register("  ") == AdjustmentResult(detail="  ", registered=False).observation()
    assert len(side_effects) == antes


async def test_k_y_el_tamano_de_chunk_llegan_al_splitter() -> None:
    sistema = construir_rag_local(k=2, chunk_size=400, chunk_overlap=40)
    docs = await sistema.obtener_top_k("vacaciones antigüedad")
    assert sistema.k == 2
    assert len(docs) <= 2
    assert sistema.retriever.retrievers[0].k == 2
    assert sistema.retriever.retrievers[1].search_kwargs["k"] == 2
    assert docs
