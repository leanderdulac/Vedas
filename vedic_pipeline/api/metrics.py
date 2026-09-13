"""Módulo de observabilidade e métricas compatível com o formato Prometheus."""

from __future__ import annotations

import threading
import time
from collections import defaultdict


class MetricsCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        # {(method, path, status): count}
        self.http_requests: dict[tuple[str, str, int], int] = defaultdict(int)
        # {(method, path): sum_duration, count}
        self.http_duration: dict[tuple[str, str], float] = defaultdict(float)
        # {backend: count}
        self.search_requests: dict[str, int] = defaultdict(int)
        # {backend: sum_duration}
        self.search_duration: dict[str, float] = defaultdict(float)
        # Gauges
        self.start_time: float = time.time()

    def record_http_request(self, method: str, path: str, status: int, duration_s: float) -> None:
        # Agrupa caminhos dinâmicos (ex.: /api/v1/documents/123 -> /api/v1/documents/:id)
        norm_path = self._normalize_path(path)
        with self._lock:
            self.http_requests[(method.upper(), norm_path, status)] += 1
            self.http_duration[(method.upper(), norm_path)] += duration_s

    def record_search(self, backend: str, duration_s: float) -> None:
        with self._lock:
            self.search_requests[backend] += 1
            self.search_duration[backend] += duration_s

    def _normalize_path(self, path: str) -> str:
        """Colapsa segmentos dinâmicos para manter cardinalidade de rótulos limitada.

        Sem isso, cada id de documento/verso viraria um rótulo único no Prometheus
        (ex.: /api/v1/verses/RV.10.129.1/audio), causando crescimento de memória e
        cardinalidade explodida no scrape.
        """
        if path.startswith("/api/v1/documents/"):
            return "/api/v1/documents/:id"
        if path.startswith("/api/v1/traditions/"):
            return "/api/v1/traditions/:id"
        if "/api/v1/verses/" in path:
            rest = path.split("/api/v1/verses/", 1)[1]
            tail = ":id"
            if rest and "/" in rest:
                _verse_id, action = rest.split("/", 1)
                tail = f":id/{action}"
            return f"/api/v1/verses/{tail}"
        return path

    def reset(self) -> None:
        with self._lock:
            self.http_requests.clear()
            self.http_duration.clear()
            self.search_requests.clear()
            self.search_duration.clear()
            self.start_time = time.time()

    def render_prometheus(self, doc_count: int = 0, chunk_count: int = 0) -> str:
        lines: list[str] = [
            "# HELP vedas_uptime_seconds Tempo de atividade da aplicação em segundos.",
            "# TYPE vedas_uptime_seconds gauge",
            f"vedas_uptime_seconds {time.time() - self.start_time:.2f}",
            "",
            "# HELP vedas_corpus_documents_total Total de documentos carregados no corpus.",
            "# TYPE vedas_corpus_documents_total gauge",
            f"vedas_corpus_documents_total {doc_count}",
            "",
            "# HELP vedas_corpus_chunks_total Total de chunks indexados.",
            "# TYPE vedas_corpus_chunks_total gauge",
            f"vedas_corpus_chunks_total {chunk_count}",
            "",
            "# HELP vedas_http_requests_total Total de requisições HTTP recebidas.",
            "# TYPE vedas_http_requests_total counter",
        ]

        with self._lock:
            for (method, path, status), count in sorted(self.http_requests.items()):
                lines.append(
                    f'vedas_http_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}'
                )

            lines.extend([
                "",
                "# HELP vedas_http_request_duration_seconds_total Tempo acumulado de requisições HTTP.",
                "# TYPE vedas_http_request_duration_seconds_total counter",
            ])
            for (method, path), total_dur in sorted(self.http_duration.items()):
                lines.append(
                    f'vedas_http_request_duration_seconds_total{{method="{method}",path="{path}"}} {total_dur:.4f}'
                )

            lines.extend([
                "",
                "# HELP vedas_search_requests_total Total de buscas semânticas por backend.",
                "# TYPE vedas_search_requests_total counter",
            ])
            for backend, count in sorted(self.search_requests.items()):
                lines.append(f'vedas_search_requests_total{{backend="{backend}"}} {count}')

            lines.extend([
                "",
                "# HELP vedas_search_duration_seconds_total Tempo acumulado de processamento de buscas.",
                "# TYPE vedas_search_duration_seconds_total counter",
            ])
            for backend, total_dur in sorted(self.search_duration.items()):
                lines.append(f'vedas_search_duration_seconds_total{{backend="{backend}"}} {total_dur:.4f}')

        lines.append("")
        return "\n".join(lines)


# Singleton
METRICS = MetricsCollector()
