# Guia de Deploy em Produção — Veda Knowledge

Este documento descreve o procedimento de deploy, configuração de ambiente, monitoramento e rotinas de manutenção para o **Vedas** em servidores Linux (Ubuntu 22.04/24.04 LTS ou Debian 12) usando Docker e Docker Compose.

---

## 1. Requisitos do Servidor

- **CPU**: 2 vCPUs ou superior (4 vCPUs recomendado para inferência e re-ranqueamento local)
- **Memória RAM**: Mínimo 4 GB (8 GB recomendado)
- **Armazenamento**: 20 GB de disco SSD/NVMe
- **Software**: Docker Engine 24+ e Docker Compose V2

---

## 2. Preparação do Ambiente

### 2.1. Clonagem e Configuração do `.env`

```bash
git clone https://github.com/leanderdulac/Vedas.git /opt/vedas
cd /opt/vedas

cp .env.example .env
```

Edite o `/opt/vedas/.env` com valores fortes de produção:

```env
# Banco de dados (senha alfanumérica; caracteres especiais exigem URL-encode)
POSTGRES_USER=vedas_prod
POSTGRES_PASSWORD=GereUmaSenhaForteAlfanumericaComPeloMenos32Caracteres
POSTGRES_DB=vedas

# Chaves de API e Tokens de Segurança
VEDIC_PIPELINE_API_TOKEN=TokenAleatorioParaOperacoesHttpDoPipeline
VEDIC_GENERATION_API_TOKEN=TokenAleatorioParaGeracaoHttp
XAI_API_KEY=xai-sua-chave-opcional

# Domínio e CORS
DOMAIN=vedas.sua-instituicao.org
CORS_ORIGINS=https://vedas.sua-instituicao.org
```

---

## 3. Inicialização dos Serviços

### 3.1. Subida dos Containers em Produção

```bash
# Sobe banco pgvector e API full-stack
docker compose -f docker-compose.prod.yml up -d

# Para subir com ingress Caddy (HTTPS automático na porta 80 e 443):
docker compose -f docker-compose.prod.yml --profile ingress up -d
```

### 3.2. Carga Inicial do Catálogo e Índices

Na primeira execução, execute os comandos do pipeline dentro do container da API:
```bash
# 1. Inicializar tabelas e extensão vector no PostgreSQL
docker compose -f docker-compose.prod.yml exec api vedic-pipeline db-init
# Com dimensão explícita (default 384; deve coincidir com VEDIC_EMBEDDING_DIM e o modelo):
# docker compose -f docker-compose.prod.yml exec api vedic-pipeline db-init --dim 768

# 2. Sincronizar os documentos do corpus com o banco
docker compose -f docker-compose.prod.yml exec api vedic-pipeline db-sync

# 3. Gerar os índices de embeddings (pgvector e numpy)
docker compose -f docker-compose.prod.yml exec api vedic-pipeline build-index --backend both
# Com dimensão explícita:
# docker compose -f docker-compose.prod.yml exec api vedic-pipeline build-index --backend both --dim 768
```

Ops pesadas também têm variante assíncrona via HTTP (`POST .../async` → 202 com
`job_id`, acompanhe com `GET /jobs/{id}`), útil para `build-index`/`train` sem
estourar timeout. Mesma autenticação `VEDIC_PIPELINE_API_TOKEN`.

### 3.3. Treino Isolado (perfil `train`)

A imagem da API é slim (sem deps de treino). Use o worker de treino:

```bash
docker compose -f docker-compose.prod.yml --profile train run --rm train \
  vedic-pipeline train-model --max-steps 10
```

O serviço `train` compartilha os volumes de dados/artefatos e a rede interna
do banco; é efêmero (`restart: "no"`, comando `sleep infinity` para `run`).

### 3.4. Object Store S3/MinIO (perfil `s3`, opcional)

Backup/restauração de `data/raw` e `artifacts/` (operação segue local-first):

```bash
# .env: VEDIC_S3_BUCKET=vedas, VEDIC_S3_ENDPOINT=http://minio:9000,
# MINIO_ROOT_USER / MINIO_ROOT_PASSWORD fortes
docker compose -f docker-compose.prod.yml --profile s3 up -d minio

docker compose -f docker-compose.prod.yml exec api vedic-pipeline artifacts status
docker compose -f docker-compose.prod.yml exec api vedic-pipeline artifacts push --dir artifacts
docker compose -f docker-compose.prod.yml exec api vedic-pipeline artifacts push --dir data/raw --prefix raw
```

O MinIO de produção não publica portas (só rede interna). Sem `VEDIC_S3_BUCKET`,
os comandos informam backend desabilitado. A API slim não inclui `boto3`;
rode `artifacts` pela CLI local (`pip install -e ".[s3]"`) ou pelo `train`.

---

## 4. Observabilidade e Monitoramento

### 4.1. Endpoint Prometheus (`/metrics`)

A API expõe métricas no formato Prometheus. Por segurança, `/metrics` **não** é
publicado pelo Caddy — scrape dentro da rede interna do compose ou via exec:

```bash
# dentro da rede interna (apenas outro container/agente em vedas_internal):
#   curl http://api:8000/metrics

# ou a partir do host:
docker compose -f docker-compose.prod.yml exec api curl -s http://127.0.0.1:8000/metrics
```

Para expor externamente, adicione `basic_auth`/rede restrita no Caddy antes de
rotear `/metrics`.

Métricas disponíveis:
- `vedas_uptime_seconds`: Tempo de atividade do processo.
- `vedas_corpus_documents_total`: Quantidade total de documentos canônicos.
- `vedas_corpus_chunks_total`: Quantidade de chunks indexados.
- `vedas_http_requests_total{method, path, status}`: Contador de requisições por rota e status HTTP.
- `vedas_http_request_duration_seconds_total{method, path}`: Latência acumulada de requisições.
- `vedas_search_requests_total{backend}`: Contagem de buscas por backend (`numpy` / `pgvector`).
- `vedas_search_duration_seconds_total{backend}`: Latência acumulada de buscas semânticas.

### 4.2. Logs Estruturados em JSON

No ambiente de produção (`docker-compose.prod.yml`), a variável `VEDIC_LOG_FORMAT=json` está ativa por padrão.
Para inspecionar logs no formato JSON:
```bash
docker compose -f docker-compose.prod.yml logs -f api
```

---

## 5. Rotinas de Backup e Manutenção

### 5.1. Backup do Banco de Dados PostgreSQL + pgvector

```bash
#!/bin/bash
BACKUP_DIR="/var/backups/vedas"
mkdir -p "$BACKUP_DIR"
DATE=$(date +%Y%m%d_%H%M%S)

docker compose -f /opt/vedas/docker-compose.prod.yml exec -T db \
  pg_dump -U vedas_prod vedas | gzip > "$BACKUP_DIR/vedas_backup_${DATE}.sql.gz"

echo "Backup concluído: $BACKUP_DIR/vedas_backup_${DATE}.sql.gz"
```

### 5.2. Restauração de Backup

```bash
gunzip -c /var/backups/vedas/vedas_backup_EXEMPLO.sql.gz | \
  docker compose -f /opt/vedas/docker-compose.prod.yml exec -T db \
  psql -U vedas_prod -d vedas
```

---

## 6. Atualização de Versão

```bash
cd /opt/vedas
git pull origin main

# Recompilação da imagem com nova versão
docker compose -f docker-compose.prod.yml build api
docker compose -f docker-compose.prod.yml up -d --no-deps api
```
