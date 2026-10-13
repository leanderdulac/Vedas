// Limpeza leve, na exibição, dos marcadores de markdown/chatbot que o modelo
// ainda emite durante o streaming. O backend devolve no evento "done" a versão
// limpa completa (vedic_pipeline/common/style.py); aqui só evitamos que o
// leitor veja asteriscos e cerquilhas enquanto o texto chega.

const HEADING = /^\s{0,3}#{1,6}\s*/;
const RULE = /^\s*(?:[-*_]\s*){3,}$/;
const BULLET = /^(\s*)(?:[-*+•▪◦]|\d{1,2}[.)])\s+(?=\S)/;
const BOLD = /(\*\*|__)(?=\S)(.+?)(?<=\S)\1/g;
const ITALIC = /(?<![*\w])\*(?=[^\s*])([^*\n]+?)(?<=[^\s*])\*(?![*\w])/g;
const DASH = /\s*(?:—|\s–\s|\s--\s)\s*/g;

export function displayProse(text: string): string {
  if (!text) return "";
  return text
    .split("\n")
    .map((line) => {
      if (RULE.test(line)) return "";
      let out = line.replace(HEADING, "").replace(/^\s*>\s?/, "").replace(BULLET, "$1");
      out = out.replace(BOLD, "$2").replace(ITALIC, "$1").replace(/\*\*/g, "");
      out = out.replace(/^\s*[—–]\s*/, "").replace(DASH, ", ").replace(/,\s*([.,;:!?)])/g, "$1");
      return out.trimEnd();
    })
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

const LOCATOR_LINE = /^\s*(?:RV|AV|SV|YV|BG|YS)\s[\d.–-]+\s*$/gm;

// Texto para a voz do navegador: sem citações [n], sem linhas de localizador
// (seriam soletradas) e sem símbolos que a síntese leria literalmente.
export function speechProse(text: string): string {
  return displayProse(text)
    .replace(/\s*\[\d+(?:\s*[,–-]\s*\d+)*\]/g, "")
    .replace(LOCATOR_LINE, "")
    .replace(/[*#_`~|<>«»“”"]/g, "")
    .replace(/\s*\n\s*/g, "\n")
    .replace(/[ \t]+/g, " ")
    .trim();
}
