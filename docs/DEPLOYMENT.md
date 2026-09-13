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
# Banco de dados
POSTGRES_USER=vedas_prod
POSTGRES_PASSWORD=GereUmaSenhaForteComPeloMenos32CaracteresAqui!
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

# 2. Sincronizar os documentos do corpus com o banco
docker compose -f docker-compose.prod.yml exec api vedic-pipeline db-sync

# 3. Gerar os índices de embeddings (pgvector e numpy)
docker compose -f docker-compose.prod.yml exec api vedic-pipeline build-index --backend both
```

---

## 4. Observabilidade e Monitoramento

### 4.1. Endpoint Prometheus (`/metrics`)

A API expõe nativamente métricas no formato padrão do Prometheus:
```bash
curl http://localhost:8000/metrics
```

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
