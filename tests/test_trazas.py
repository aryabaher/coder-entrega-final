"""Cinco ejecuciones, cinco trazas. El exporter en memoria hace de Phoenix cuando TRAZAS_OFFLINE=1."""

from app import worker
from app.graph import run_graph
from app.observability import memory

PREGUNTAS = [
    "¿Cuántos días de vacaciones corresponden con 5 años de antigüedad?",
    "¿Cuántos días de teletrabajo por semana permite la política?",
    "¿Qué tiene que completar una persona nueva durante el onboarding?",
    "¿Cuál es la regla de contraseñas de la política de seguridad informática?",
    "¿Cuántos días de vacaciones acumula en total un empleado en sus primeros 12 años?",
]


async def test_cinco_ejecuciones_dejan_cinco_trazas() -> None:
    memory.clear()
    for indice, pregunta in enumerate(PREGUNTAS):
        hilo = f"traza-{indice}"
        resultado = await worker._run_traced(
            {"job_id": f"job-{indice}", "thread_id": hilo},
            pregunta,
            run_graph,
            pregunta,
            {"configurable": {"thread_id": hilo}},
            25,
        )
        assert resultado["paused"] is False
        assert resultado["result"]
    raices = [span for span in memory.get_finished_spans() if span.name == "orquestador"]
    assert len(raices) == 5
    assert {span.attributes["session.id"] for span in raices} == {f"traza-{i}" for i in range(5)}
    assert all(span.attributes["input.value"] == pregunta for span, pregunta in zip(raices, PREGUNTAS, strict=True))
