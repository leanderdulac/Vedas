# Estilo das imagens (figuras e versos)

Referência: um acervo local de ~360 imagens devocionais que o Leandro juntou
(arte de IA, impressos, fotos de templo). Elas NÃO entram no git, porque são
imagens de terceiros baixadas do Facebook e o direito autoral é incerto; só
podem ser apontadas por um caminho local no `.env`.

| Grupo do acervo | Qtd. aprox. | Uso no app |
| --- | --- | --- |
| Devocional cinematográfico de IA (raios volumétricos, halo dourado, céu dramático) | ~145 | **stills de verso** (`VEDIC_SCENE_STYLE=cinematic`) |
| Devocional de IA brilhante e polido (lótus, mandala, frontal) | ~105 | — |
| Pintura tradicional e impressos (Ravi Varma, calendário, ISKCON, Pahari) | ~60 | `miniature` (legado) |
| Fotos de mūrtis e liṅgas | ~27 | — |
| Série escura e escultural de devas (Agni, Brahmā, Indra, Vāyu, Yama…) | 9 | **retratos** (`VEDIC_FIGURE_STYLE=sculpted`) |

## Retrato (`sculpted`, padrão)
Busto de cabeça e ombros, centralizado, como uma escultura viva de bronze e
obsidiana: filigrana gravada, paisley e ornamento tribal na pele e na roupa,
coroa pesada, brincos e colares em camadas, olhar intenso, chiaroscuro com luz
de recorte, fundo quase preto e enfumaçado, paleta de bronze e carvão com uma
cor de destaque. A iconografia sempre vem antes do estilo.

## Verso (`cinematic`, padrão)
Pintura digital devocional semi-realista, raios volumétricos e halo dourado,
céu dramático, paleta de ouro, açafrão e azul profundo, composição heroica
centralizada. A imagem também serve de primeiro quadro do vídeo.

## O que aprendemos no teste (grok-imagine-image-quality, 2026-10-02)
- Escrever "seven flaming tongues" fazia o modelo desenhar bigodes de fogo. A
  versão "crowns of living flame with seven tongues of fire rising above
  them" resolve.
- "noose" vira forca; o texto agora pede "pasha, a coiled lasso of glowing
  light".
- No `/images/edits`, as referências de estilo copiam também a composição:
  o Agni saiu com três cabeças em 4 de 4 imagens, mesmo com referências de uma
  cabeça só. Por isso `VEDIC_IMAGE_STYLE_REF` e `VEDIC_SCENE_STYLE_REF`
  existem, mas ficam vazios por padrão.
- Os estilos novos têm cara de render de propósito, então "photorealistic" e
  "3d render" só entram no "Avoid" quando o estilo é `miniature`.

| Variável | Padrão | Efeito |
| --- | --- | --- |
| `VEDIC_FIGURE_STYLE` | `sculpted` | `sculpted`, `cinematic` ou `miniature` |
| `VEDIC_SCENE_STYLE` | `cinematic` | `cinematic` ou `miniature` |
| `VEDIC_IMAGE_STYLE_REF` | vazio | Pasta ou arquivos (separados por `:`) de referência para retratos; usa até 2 e ignora o arquivo com o nome do próprio personagem |
| `VEDIC_SCENE_STYLE_REF` | vazio | O mesmo, para os stills de verso |

Com referência, a chamada vai para `/images/edits` (cerca de US$ 0,06 por
imagem, contra US$ 0,05 e uns 4 s a menos sem ela). Se a edição falhar, a
imagem é gerada só pelo texto.
