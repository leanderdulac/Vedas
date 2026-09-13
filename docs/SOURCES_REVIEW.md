# Fontes de referência e integração — Veda Knowledge

Análise de cada URL indicada quanto a **licença**, **formato**, **encaixe no
pipeline atual** e **recomendação de adoção**. O projeto tem um gate jurídico
rígido (`ALLOWED_LICENSES` em `vedic_pipeline/common/constants.py`): só entra
no corpus o que tem licença explícita permitida e `source_id`/`fingerprint`
estáveis. Os textos sagrados em si são obras antigas (domínio público), mas a
*transcrição digital* e as *traduções modernas* têm copyright — é isso que o
gate avalia.

Resumo executivo: só uma fonte vale **adoção direta agora** (o portal do
governo da Índia). As demais são referências, já cobertas, ou incompatíveis
com a redistribuição MIT/CC do projeto.

---

## 1. Vedic Heritage Portal — [vedicheritage.gov.in](https://vedicheritage.gov.in/)

- **O que é:** portal oficial do Ministério da Cultura da Índia. Texto sânscrito
  (Devanāgarī) de **todos os quatro Vedas** (Rig, Yajur Shukla/Kṛṣṇa, Sāma,
  Atharva), Brāhmaṇas, Āraṇyakas e as principais Upaniṣads, organizado por
  saṃhitā/tradição ([Samhitas](https://vedicheritage.gov.in/samhitas),
  [Upanishads](https://vedicheritage.gov.in/upanishads/)).
- **Licença:** conteúdo governamental reutilizável **com atribuição**
  ([Copyright Policy](https://vedicheritage.gov.in/copyright-policy/), [Termos](https://vedicheritage.gov.in/terms-conditions/)).
  As escrituras são obras antigas; a digital reproduz edições acadêmicas. Uso
  educacional com citação da fonte é o cenário esperado.
- **Formato / encaixe:** HTML por página (WordPress); exigiria um adaptador novo
  de scraping no lugar de `extract_html`. O pipeline já tem parsers estruturais
  para Ṛgveda/YV/SV/AV (`etl/structure.py`, `etl/structured_json.py`); os
  textos do portal são longos o suficiente para `parse_rigveda`/`parse_numbered_verses`.
- **Recomendação: ✅ ADOTAR (prioridade alta).** É a melhor fonte única para
  *completar a cobertura da śruti* além do Rig (Yajur/Atharva/Sāma, Brāhmaṇas,
  Upaniṣads). Plano concreto:
  1. Adicionar token de licença `gov-ind` (já incluído em `ALLOWED_LICENSES`);
  2. Criar `vedic_pipeline/etl/vedicheritage.py` que baixa as páginas listadas
     (respeitando `vedic_pipeline/crawler/download.py` / `is_safe_url` para SSRF),
     extrai o Devanāgarī e reutiliza `parse_document_units`;
  3. Criar um manifesto em `fixtures/` apontando as URLs por saṃhitā com
     `"license": "gov-ind"` e `"attribution"` para o portal;
  4. Rodar `ingest` + `dedupe` + `build-index`.
  **Atenção:** confirmar no `copyright-policy` a permissão para redistribuir em
  projeto aberto antes de publicar o corpus derivado; é aceitável para a fase
  de índice local/RAG.

  **Status (implementado):**
  - `vedic_pipeline/etl/vedicheritage.py` — extrai o Devanāgarī canônico do
    bloco `#videotext` (não-modal, maior conteúdo), monta registro de corpus
    (`language=sa`, `license=gov-ind`, `attribution`), reutilizando SSRF/`gov-ind`.
  - Rota em `crawler/ingest.py`: `source_class: "vedicheritage"` (ou host do
    portal) aciona o adaptador.
  - `scripts/build_vedicheritage_manifest.py` — gera (a partir do sitemap
    oficial) o **manifesto completo das saṃhitās**: `fixtures/sources_vedicheritage.json`
    com **1860 fontes** de texto (Ṛgveda 1011 sūktas, Yajurveda 118 capítulos,
    Atharvaveda Śaunaka 731 sūktas) — cada página vira um registro de corpus.
    Regenere com `python scripts/build_vedicheritage_manifest.py`.
  - `fixtures/vedicheritage/isha_upanishad.html` (amostra p/ testes) e
    `tests/test_vedicheritage.py`; validado também contra HTML real do portal.
  - **Nota:** o Sāma Veda não expõe páginas de texto no sitemap (apenas menus
    de recensão), então fica de fora até o portal publicar o texto.


---

## 2. VedaWeb (legacy) — [vedaweb-legacy](https://github.com/VedaWebProject/vedaweb-legacy)

- **O que é:** repositório **arquivado** com o código da plataforma antiga do
  VedaWeb (Elasticsearch/MongoDB). O README aponta explicitamente para o
  software-sucessor `VedaWebProject/Tekst` e para os **dados** em
  `VedaWebProject/vedaweb-data`.
- **Licença:** o dado relevante (camada de texto ISO‑15919 de Zurique, ṛgveda)
  é **CC BY 4.0** — a mesma que o projeto **já integra** em
  `vedic_pipeline/etl/vedaweb.py` e `scripts/import_vedaweb.py`.
- **Recomendação: ⚠️ NÃO ADOTAR novos.** O legacy é o código da plataforma (não
  há dado novo); o corpus de texto já está coberto. As camadas de morfologia e
  traduções do VedaWeb são NC (o projeto deliberadamente as exclui). O único
  ganho seria apontar a API ([document](https://vedaweb.uni-koeln.de/rigveda/openapi))
  como referência — registrar em docs.

---

## 3. Library of Sacred Vedic Texts — [vedicfriends.org](https://www.vedicfriends.org/library_of_sacred_vedic_texts.htm)

- **O que é:** **apenas um diretório de links** ("Library ... is still being put
  together…"). Não hospeda texto; aponta para sacred-texts, Project Gutenberg,
  ISKCON e outros. Muitos links antigos/mortos.
- **Recomendação: ⚠️ NÃO É FONTE DE DADOS.** Nada a ingerir. No máximo, servir
  como inventário de URLs públicas ao montar manifestos. As traduções listadas
  (Griffith, Prabhupada…) têm licenças distintas — avaliar individualmente.

---

## 4. HinduScriptures (Jayesh Patel) — [repository](https://github.com/jayeshmepani/HinduScriptures)

- **O que é:** repositório "Digital Repository & AI Scholar" de escrituras.
- **Licença:** **GPL v3** para o pacote/reposição ([LICENSE](https://github.com/jayeshmepani/HinduScriptures/blob/main/LICENSE)).
- **Recomendação: ⛔ NÃO INGERIR no corpus.** O projeto é MIT e redistribui o
  corpus em CC/PD; incorporar dados sob **GPL‑v3 contamina a redistribuição**
  (copyleft da compilação). O texto sânscrito em si é domínio público, mas o
  *dataset empacotado* está sob GPL. Se necessário, usar apenas a identidade de
  verse_id como referência de conferência — nunca o conteúdo derivado.

---

## 5. Bhagavad-gita API — [vedicscriptures/bhagavad-gita-api](https://github.com/vedicscriptures/bhagavad-gita-api)

- **O que é:** **API REST** JSON com os 700 ślokas e 18 capítulos da Gītā.
- **Licença:** **GPL v3** ([LICENSE](https://github.com/vedicscriptures/bhagavad-gita-api/blob/main/LICENSE)) e é um serviço ao vivo (sem export em massa).
- **Recomendação: ⛔ NÃO INGERIR (GPL + API).** A Gītā já está coberta no corpus
  (DharmicData ODbL, sacred-texts/Griffith PD). Manter apenas como referência
  externa. O conteúdo sânscrito é PD, mas o dataset compilado é GPL.

---

## 6. Veducation — Sastra / Gita — [página](https://www.veducation.world/library/Sastra-%E0%A4%B6%E0%A4%BE%E0%A4%B8%E0%A5%8D%E0%A4%A4%E0%A5%8D%E0%A4%B0/gita/bhagavad-gita/%E0%A4%AD%E0%A4%97%E0%A4%B5%E0%A4%A6%E0%A5%8D%E0%A4%97%E0%A5%80%E0%A4%A4%E0%A4%BE--%E0%A4%97%E0%A5%80%E0%A4%A4%E0%A4%BE%E0%A4%AA%E0%A5%8D%E0%A4%B0%E0%A5%87%E0%A4%B8)

- **O que é:** página web com Gītā (transliteração/IAST provavelmente).
- **Licença:** não declarada de forma legível/machine‑readable na página.
- **Recomendação: ⚠️ Referência apenas.** Sem licença verificável e sem export
  estruturado, não entra. A Gītā já é coberta por outras fontes.

---

## 7. Sanskrit Documents — [sanskritdocuments.org](https://sanskritdocuments.org/)

- **O que é:** extenso acervo em Devanāgarī e IAST (Vedas, Upaniṣads, Purāṇas,
  itihāsa, stotras), base voluntária desde os anos 90.
- **Licença:** **política de cópia restritiva/consciência** ([FAQ](https://sanskritdocuments.org/faq)):
  desaconselha cópia em massa para promoção/comercio; exige **atribuição,
  link para a origem e permissão/consciência**. Não é uma licença aberta
  compatível com redistribuição automática em corpus MIT/CC.
- **Formato:** muitos arquivos em **ITRANS** (precisaria transliterar), outros em
  Devanāgarī; os textos-base vêm de livros em domínio público.
- **Recomendação: ⚠️ usar como REFERÊNCIA / conferência cruzada, não ingerir
  em massa.** Para adoção, pedir permissão explícita e registrar atribuição.
  Alternativa limpa para os mesmos textos PD: mirrors já licenciados
  (sacred-texts, sa.wikisource).

---

## 8. Online Darshan — Gita — [onlinedarshan.com/gita](https://www.onlinedarshan.com/gita/onlinegita.asp?id=1)

- **O que é:** interface web da Gītā (renderização em páginas ASP).
- **Licença / formato:** não é export em massa nem machine-readable; licença não
  declarada.
- **Recomendação: ⚠️ Referência apenas.** Sem valor de ingestão.

---

## Matriz de decisão

| Fonte | Licença | Formato | Adotar? | Ação |
|---|---|---|---|---|
| vedicheritage.gov.in | gov‑ind (atribuição) | HTML/Devanāgarī | ✅ sim (alta) | adaptador + manifesto + `gov-ind` (token já adicionado) |
| vedaweb‑legacy | CC BY 4.0 (dado Zürich) | já integrado | ⚠️ não (arquivado/coberto) | link de referência |
| vedicfriends.org | n/a (links) | link list | ⛔ não | inventário de URLs |
| HinduScriptures (jayeshmepani) | GPL‑v3 | dataset | ⛔ não | referência de verse_id apenas |
| bhagavad‑gita‑api | GPL‑v3 + API | JSON/API | ⛔ não | referência externa |
| veducation Gita | indefinida | página | ⚠️ não | referência |
| sanskritdocuments.org | uso educ. c/ atribuição | Devanāgarī/ITRANS | ⚠️ não (pedir permissão) | referência/conferência |
| onlinedarshan Gita | indefinida | página | ⚠️ não | referência |

## Próximos passos sugeridos

1. **Página Gita (Veducation/onlinedarshan) e repositórios GPL:** nenhuma ação
   além de documentar (feito acima).
2. **Vedic Heritage:** implementar o adaptador e o manifesto (fase 1 da seção 1)
   — é a única fonte que amplia o corpus hoje. Validar o `copyright-policy` para
   redistribuição aberta antes de publicar o corpus derivado.
3. **sanskritdocuments.org:** apenas se houver interesse em stotras/itihāsa com
   permissão explícita; para os Vedas prefira vedicheritage.

*Nota: dados obtidos via web em 2026‑09‑07; termos podem mudar — reconfirme as
políticas na hora de integrar.*
