"""Módulo de observabilidade e métricas compatível com o formato Prometheus."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any


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
                # Allowlist de sufixos conhecidos — resto vira :id/action genérico.
                first = action.split("/", 1)[0]
                safe = first if first in {"explain", "audio", "image", "video"} else "other"
                tail = f":id/{safe}"
            return f"/api/v1/verses/{tail}"
        # Capta apenas paths conhecidos; resto colapsa para evitar injeção/cardinalidade.
        known = {
            "/metrics", "/health", "/ask", "/search", "/ingest", "/tokenize",
            "/train", "/build-index", "/db/init", "/db/sync",
            "/api/v1/health", "/api/v1/stats", "/api/v1/traditions",
            "/api/v1/documents", "/api/v1/search", "/api/v1/ask",
            "/api/v1/ask/stream", "/api/v1/media/cached",
        }
        if path in known:
            return path
        if path.startswith("/api/v1/"):
            return "/api/v1/other"
        if path.startswith("/assets/"):
            return "/assets/file"
        return "/other"

    @staticmethod
    def _escape_label(value: str) -> str:
        return value.replace("\\", r"\\").replace('"', r"\"").replace("\n", r"\n")

    def reset(self) -> None:
        with self._lock:
            self.http_requests.clear()
            self.http_duration.clear()
            self.search_requests.clear()
            self.search_duration.clear()
            self.start_time = time.time()

    def render_prometheus(
        self,
        doc_count: int = 0,
        chunk_count: int = 0,
        job_counts: dict[str, Any] | None = None,
    ) -> str:
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

        if job_counts:
            lines.extend(
                [
                    "",
                    "# HELP vedas_jobs_total Jobs do pipeline por status.",
                    "# TYPE vedas_jobs_total gauge",
                    f"vedas_jobs_total {int(job_counts.get('total') or 0)}",
                    "",
                    "# HELP vedas_jobs_by_status_total Jobs do pipeline agrupados por status.",
                    "# TYPE vedas_jobs_by_status_total gauge",
                ]
            )
            for status, count in sorted((job_counts.get("by_status") or {}).items()):
                s = self._escape_label(str(status))
                lines.append(f'vedas_jobs_by_status_total{{status="{s}"}} {int(count)}')
            lines.extend(
                [
                    "",
                    "# HELP vedas_jobs_by_kind_total Jobs do pipeline agrupados por tipo.",
                    "# TYPE vedas_jobs_by_kind_total gauge",
                ]
            )
            for kind, count in sorted((job_counts.get("by_kind") or {}).items()):
                k = self._escape_label(str(kind))
                lines.append(f'vedas_jobs_by_kind_total{{kind="{k}"}} {int(count)}')

        with self._lock:
            for (method, path, status), count in sorted(self.http_requests.items()):
                m = self._escape_label(method)
                p = self._escape_label(path)
                lines.append(
                    f'vedas_http_requests_total{{method="{m}",path="{p}",status="{status}"}} {count}'
                )

            lines.extend([
                "",
                "# HELP vedas_http_request_duration_seconds_total Tempo acumulado de requisições HTTP.",
                "# TYPE vedas_http_request_duration_seconds_total counter",
            ])
            for (method, path), total_dur in sorted(self.http_duration.items()):
                m = self._escape_label(method)
                p = self._escape_label(path)
                lines.append(
                    f'vedas_http_request_duration_seconds_total{{method="{m}",path="{p}"}} {total_dur:.4f}'
                )

            lines.extend([
                "",
                "# HELP vedas_search_requests_total Total de buscas semânticas por backend.",
                "# TYPE vedas_search_requests_total counter",
            ])
            for backend, count in sorted(self.search_requests.items()):
                b = self._escape_label(backend)
                lines.append(f'vedas_search_requests_total{{backend="{b}"}} {count}')

            lines.extend([
                "",
                "# HELP vedas_search_duration_seconds_total Tempo acumulado de processamento de buscas.",
                "# TYPE vedas_search_duration_seconds_total counter",
            ])
            for backend, total_dur in sorted(self.search_duration.items()):
                b = self._escape_label(backend)
                lines.append(f'vedas_search_duration_seconds_total{{backend="{b}"}} {total_dur:.4f}')

        lines.append("")
        return "\n".join(lines)


# Singleton
METRICS = MetricsCollector()
