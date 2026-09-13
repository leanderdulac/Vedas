# SexyAds e Vedas — análise e desenvolvimento local

Data: 7 de setembro de 2026. Revisão inicial de arquitetura, autenticação, APIs, persistência, busca, clientes web, testes e execução. Não é uma auditoria exaustiva de todas as funcionalidades.

## Resultado disponível neste computador

| Projeto | Endereço | Estado |
|---|---|---|
| SexyAds | http://localhost:3100 | Interface compilada; 25 anúncios de demonstração |
| SexyAds API | http://localhost:3101/health | PostgreSQL e Redis conectados |
| Vedas | http://localhost:8100 | Biblioteca, busca e respostas extrativas com fontes e streaming |

Código em `/Users/leandrofrancademello/Projetos/SexyAds` e `/Users/leandrofrancademello/Projetos/Vedas`. Alterações locais, sem commit, push ou publicação externa.

O SexyAds usa os novos containers `sexyads-local-postgres` (PostGIS, porta 15433, volume `sexyads_local_postgres`) e `sexyads-local-redis` (porta 16379), ligados apenas ao loopback. Banco de desenvolvimento `sexyads`, banco separado de testes `sexyads_test`. Não foram usados os bancos dos outros projetos.

O Vedas usa Python 3.12 em `.venv`, corpus local de **11 documentos / 45 chunks / 384 dimensões**, embeddings multilingues e backend NumPy. A porta 8000 já pertencia a outro processo, por isso a aplicação foi iniciada em 8100. As 1.054 fontes citadas no README original são um snapshot de outro ambiente; não estão presentes neste clone.

## SexyAds: correções implementadas

1. **Renovação de sessão concorrente:** duas requisições podiam ler o mesmo refresh token antes da revogação e ambas emitir sucessores. Agora o consumo usa atualização condicional dentro da transação; apenas uma requisição vence. Contas suspensas ou excluídas e tokens vencidos são recusados.
2. **Autorização administrativa:** `requireRole` agora reutiliza a autenticação completa, incluindo revogação. A cada requisição protegida, o estado e o papel atual da conta são consultados no banco, impedindo acesso com JWT antigo após suspensão, exclusão ou remoção do papel de administrador.
3. **Testes seguros:** a suíte exige explicitamente banco com nome terminado em `_test` e roda em série, pois as suítes compartilham tabelas e apagam fixtures. CI atualizado para banco de teste e Redis.
4. **Inicialização:** o seed carrega `.env`; anteriormente `npm run db:seed` falhava por ausência de `DATABASE_URL`, mesmo com o arquivo criado. Dependências, Prisma Client, 11 migrations e dados demo foram preparados.
5. **Execução compilada:** adicionado `npm run start:local` para iniciar API e frontend já compilados.

### Próximas prioridades propostas

| Prioridade | Evidência no código | Desenvolvimento recomendado |
|---|---|---|
| Alta | `backend/src/services/payment.service.ts`: pagamento é marcado concluído antes de ativar campanha/plano, em operações separadas | Tornar confirmação e benefícios transacionais/idempotentes; testar duplicação de webhooks e falhas intermediárias |
| Alta | `backend/src/utils/redis.ts`: revogação é ignorada se Redis estiver indisponível | Definir comportamento de falha para autenticação e persistir versão de sessão/revogação de forma durável |
| Alta | `frontend/src/lib/api.ts` e `store/auth.ts`: refresh token é salvo, mas não há renovação automática; erros de rede em `refreshMe` apagam a sessão | Implementar renovação coordenada, atualização consistente do estado e distinção entre falha de rede e sessão inválida |
| Média | Refresh tokens são armazenados em texto e validade de 30 dias está fixa no serviço | Persistir hash, respeitar configuração de validade e remover tokens expirados periodicamente |
| Média | README indicava Next.js 14, mas dependência instalada é 16.3; `lint` ainda usa `next lint` | Completar atualização de documentação e configurar lint compatível |
| Alta | `npm audit --json` retornou 11 entradas de severidade alta, incluindo dependências transitivas de Fastify e Prisma | Validar advisories e atualizar dependências em lote separado, com testes de compatibilidade; não executar atualização forçada indiscriminada |

A checagem de conta por requisição acrescenta uma consulta ao banco. É uma escolha explícita para aplicar suspensão e mudança de papel imediatamente; medir latência antes de introduzir cache.

## Vedas: correções implementadas

1. **Chunking sem travamento:** valida tamanho/sobreposição e garante avanço mesmo quando uma quebra de palavra produz trecho menor que a sobreposição. Teste com timeout impede que uma regressão congele a suíte.
2. **API validada:** consultas vazias ou apenas espaços, consultas excessivas, backends/provedores inválidos e sobreposição incompatível são rejeitados pela validação.
3. **Operações administrativas do pipeline:** ingestão, tokenização, treino, construção de índice e sincronização/inicialização de banco exigem `Authorization: Bearer …` e `VEDIC_PIPELINE_API_TOKEN`. Sem configuração, os endpoints respondem 503. A CLI continua funcionando sem token.
4. **Arquivos estáticos:** a rota de fallback resolve o caminho e impede acesso fora de `frontend/dist`, inclusive através de links simbólicos.
5. **Filtros de recuperação:** candidatos lexicais agora respeitam idioma e tradição, assim como os candidatos semânticos. Antes, o ranking híbrido reintroduzia documentos excluídos pelos filtros.
6. **Títulos compostos:** a tokenização trata hífen como separador. `Bhagavad-gita` passa a corresponder a `Bhagavad Gita`; a suíte de referência passou de 3/4 para 4/4 sem alterar as perguntas ou seus critérios.
7. **Streaming web:** parser SSE mantém estado entre fragmentos de rede e trata CRLF e múltiplas linhas. Antes, uma linha `data:` recebida antes do terminador do evento podia ser perdida. Testes cobrem todas as posições de fragmentação em dois blocos.
8. **Exibição do ranking:** o cartão mostra score numérico, em vez de percentual que podia exceder 100% e sugerir confiança estatística.
9. **Execução sem PostgreSQL:** scripts não impõem mais um banco na porta 5432; `.env` pode configurar banco e, sem banco, a API usa NumPy. CI ganhou regressões Python/API e testes/build do frontend.

### Próximas prioridades propostas

| Prioridade | Evidência no código | Desenvolvimento recomendado |
|---|---|---|
| Alta | API pública aceita seleção de diretório de índice e modelo/provedor | Restringir artefatos/modelos permitidos no servidor e autenticar/limitar requisições que possam consumir recursos ou créditos |
| Alta | Dependências Python usam apenas limites mínimos; npm audit retornou 7 entradas (3 altas e 4 moderadas) no frontend | Fixar ambiente reproduzível, separar dependências de treino das de execução e tratar advisories de Vite/roteador/transitivas |
| Média | `catalog_service.py` relê JSONL; `hybrid.py` retokeniza candidatos a cada consulta | Cache com invalidação por versão do corpus; índice lexical persistente e benchmarks com corpus completo |
| Média | Índice cacheado em processo, atualização dependente do caminho de execução | Versionar e trocar índices atomicamente; invalidar caches depois de alterações pela CLI |
| Média | `api/app.py` devolve mensagens internas de exceção e operações pesadas executam na própria requisição | Mensagens públicas controladas, logs estruturados e fila de trabalhos com progresso/cancelamento |
| Média | Corpus atual é de exemplos; métricas de ranking não são probabilidade de acerto | Ampliar fontes com procedência verificada e avaliar recall/diversidade com um conjunto independente maior |

## Vedas — revisão adicional (hardening, consistência e segurança)

Passo de revisão do código completo (backend Python + frontend React). Tudo verde: 75 testes Python + 23 subtests, `ruff check .` limpo.

### Vulnerabilidades/custos resolvidos

1. **Geração de mídia paga ficava aberta sem autenticação:** `/api/v1/verses/{id}/audio` (TTS), `/image` e `/video` (POST de início) disparavam chamadas pagas na chave xAI de qualquer anônimo, enquanto `/ask` e `/explain` já exigiam o `VEDIC_GENERATION_API_TOKEN`. Agora esses endpoints aplicam `authorize_media()` — mesma política de token do `/ask`. Em instância local/privada (token vazio) nada muda; em API pública com token, anônimos não gastam crédito. Leitura de mídia em cache (`/image`, `/video` no estado pronto) continua livre.
2. **Cardinalidade de métricas: potencial DoS de memória/Prometheus.** Cada `verse_id`/`doc_id` virava rótulo único (`/api/v1/verses/RV.10.129.1/audio`). `_normalize_path` agora colapsa segmentos dinâmicos (`/api/v1/verses/:id/:action`, `/api/v1/documents/:id`, `/api/v1/traditions/:id`).
3. **Rate limiting anti-abuso/DoS** nas rotas caras públicás (`/ask`, `/ask/stream`, `/search`, `/api/v1/verses/*`): janela fixa de 1 min por (grupo, IP). Configurável via `VEDIC_RATE_LIMIT_ENABLED` / `VEDIC_RATE_LIMIT_PER_MINUTE`. Leituras e estáticos não são limitados. Retorna `Response` 429 direto (middleware está fora do ExceptionMiddleware, então `HTTPException` viraria 500).

### Consistência

4. **Versão unificada.** Havia três números divergentes (pyproject 2.0.0, `app.py` "2.0.0" e `__init__.__version__` "1.2.0"). Agora `vedic_pipeline/__init__.__version__ = "2.0.0"` é a fonte única, consumida pela API e `/health`.
5. **`ruff check .` limpo** (3 erros de ordenação de imports corrigidos).
6. **Prod compose:** o Caddy está no perfil `ingress`; o cabeçalho de uso documenta `--profile ingress` (o comando `up -d` sozinho não subia o ingress).

### Propostas de melhoria (não alteradas)

| Prioridade | Ponto | Recomendação |
|---|---|---|
| Alta | API Docker instala `torch`/`transformers`/`datasets` (dependências de treino) num serviço só de API | Separar runtime (API+embeddings) de train; imagem muito menor |
| Alta | `npm audit` retorna advisories no frontend (Vite, react-router) | Subir e validar em lote separado |
| Média | `/metrics` exposto publicamente no Caddyfile sem auth | Restringir à rede interna ou adicionar `basic_auth` no Caddy |
| Média | Operações pesadas (build-index, train) rodam na própria requisição HTTP | Fila de trabalhos com progresso/cancelamento |
| Média | SSRF DNS check pode ser desabilitado por `VEDIC_DISABLE_SSRF_DNS_CHECK` | Mantê-lo sempre ligado em produção (só teste) |

## Validação realizada

- **SexyAds:** build TypeScript do backend e build Next.js (49 páginas geradas); 27 testes backend e 5 frontend aprovados.
- **SexyAds HTTP real:** login; duas renovações simultâneas retornando 200 e 401; consulta de sessão; logout; recusa do JWT revogado.
- **SexyAds navegador:** página inicial e anúncios renderizados. Login visual limitado pelo navegador integrado, que bloqueia acesso direto à porta 3101; o mesmo fluxo passou pela API HTTP.
- **Vedas:** 11 testes Python/API e 3 testes de streaming frontend aprovados; build Vite aprovado.
- **Vedas recuperação:** 4/4 consultas do gold set de CI e resposta extrativa aprovadas, com modelo real de embeddings. Busca HTTP com filtros retornou apenas documentos correspondentes.
- **Vedas navegador:** página inicial, navegação e pergunta com resposta extrativa, fontes e streaming concluído; sem erros de console observados.
- **Total:** 46 testes aprovados, além das 4 consultas de recuperação e verificações HTTP/navegador.

Não foram testados pagamentos reais, fornecedores externos de avatar/voz, geração xAI/local generativa, treino de modelos ou recuperação pgvector. Não há chaves desses fornecedores configuradas. Os avisos de dependências ainda precisam de tratamento. O modo `next dev` encontrou `EMFILE` no ambiente de execução do agente; a versão compilada funciona. Esses limites não foram ocultados com alterações nos critérios de teste.

## Como continuar localmente

### SexyAds

```bash
cd /Users/leandrofrancademello/Projetos/SexyAds
docker start sexyads-local-postgres sexyads-local-redis
npm run build
npm run start:local
```

Pare os processos já em execução antes de iniciar outra instância nas mesmas portas. Os arquivos `backend/.env` e `frontend/.env.local` já foram criados e são ignorados pelo Git. Conta demo: `cliente@sexyads.local`, senha `password123`.

```bash
DATABASE_URL=postgresql://sexyads:sexyads@localhost:15433/sexyads_test REDIS_PORT=16379 npm test -w backend -- --runInBand --watchman=false
npm test -w frontend -- --runInBand --watchman=false
```

Não use o seed como migração rotineira: ele apaga e recria dados demo. Foi executado somente no novo banco de desenvolvimento preparado nesta sessão.

### Vedas

```bash
cd /Users/leandrofrancademello/Projetos/Vedas
(cd frontend && npm run build)
.venv/bin/uvicorn vedic_knowledge_pipeline:app --host 127.0.0.1 --port 8100
```

`.env` aponta o cache de embeddings para `artifacts/huggingface`, já baixado, em modo offline. Para baixar outros modelos, desative `HF_HUB_OFFLINE`. A resposta atual é extrativa, sem custo de LLM externo.

```bash
.venv/bin/python -m unittest discover -s tests -v
(cd frontend && npm test)
.venv/bin/python scripts/smoke_rag.py --backend numpy --queries fixtures/smoke_queries_ci.json --strict --json-out data/smoke_report.json
```

Testes do frontend usam Node.js 22.6+ com suporte a remoção de tipos. O corpus amplo e os bancos de outros ambientes não foram importados.
