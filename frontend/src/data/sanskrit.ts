// Dados da página "Aprenda" (saṃskṛtam): varṇamālā, vocabulário védico e lições.
// Conteúdo tradicional consolidado — pontos de articulação conforme o Śikṣā
// (svaras + vyañjanas por varga), vocabulário do corpus védico em uso no app.

export type Varna = {
  deva: string;
  iast: string;
  group: string;
  articulation: string; // ponto de articulação em PT
  example?: { word: string; iast: string };
};

export type VocabEntry = {
  deva: string;
  iast: string;
  gloss: string; // glosa PT curta
  meaning: string; // significado védico (1-2 frases)
  query: string; // consulta para achar a palavra viva no corpus
  category: "deva" | "ritual" | "cosmos" | "principio";
};

export type Lesson = {
  id: string;
  title: string;
  subtitle: string;
  sections: {
    heading: string;
    body: string[];
    examples: { deva: string; iast: string; gloss: string }[];
  }[];
};

// ---------------------------------------------------------------- varṇamālā

const GROUP_LABELS: Record<string, string> = {
  svara: "Svaras (vogais)",
  kavarga: "Ka-varga — guturais",
  cavarga: "Ca-varga — palatais",
  tavarga: "Ṭa-varga — retroflexas",
  tavarga2: "Ta-varga — dentais",
  pavarga: "Pa-varga — labiais",
  antahstha: "Antaḥstha (semivogais)",
  usman: "Ūṣman (sibilantes e aspiradas)",
};

export const VARNA_GROUPS: { id: string; label: string; items: Varna[] }[] = [
  {
    id: "svara",
    label: GROUP_LABELS.svara,
    items: [
      { deva: "अ", iast: "a", group: "svara", articulation: "curta, da garganta", example: { word: "अग्नि", iast: "agni" } },
      { deva: "आ", iast: "ā", group: "svara", articulation: "longa, da garganta", example: { word: "आत्मन्", iast: "ātman" } },
      { deva: "इ", iast: "i", group: "svara", articulation: "curta, palatal", example: { word: "इन्दु", iast: "indu" } },
      { deva: "ई", iast: "ī", group: "svara", articulation: "longa, palatal", example: { word: "ईश", iast: "īśa" } },
      { deva: "उ", iast: "u", group: "svara", articulation: "curta, labial", example: { word: "उपनिषद्", iast: "upaniṣad" } },
      { deva: "ऊ", iast: "ū", group: "svara", articulation: "longa, labial", example: { word: "ऊर्जा", iast: "ūrjā" } },
      { deva: "ऋ", iast: "ṛ", group: "svara", articulation: "vogal silábica, retroflexa", example: { word: "ऋतम्", iast: "ṛtam" } },
      { deva: "ॠ", iast: "ṝ", group: "svara", articulation: "vogal silábica longa, retroflexa" },
      { deva: "ऌ", iast: "ḷ", group: "svara", articulation: "vogal silábica, dental-lateral" },
      { deva: "ॡ", iast: "ḹ", group: "svara", articulation: "vogal silábica longa, dental-lateral" },
      { deva: "ए", iast: "e", group: "svara", articulation: "ditongo, gutural-palatal", example: { word: "एक", iast: "eka" } },
      { deva: "ऐ", iast: "ai", group: "svara", articulation: "ditongo, gutural-palatal", example: { word: "ऐश्वर्य", iast: "aiśvarya" } },
      { deva: "ओ", iast: "o", group: "svara", articulation: "ditongo, gutural-labial", example: { word: "ओम्", iast: "oṃ" } },
      { deva: "औ", iast: "au", group: "svara", articulation: "ditongo, gutural-labial", example: { word: "औपनिषद", iast: "aupaniṣada" } },
    ],
  },
  {
    id: "kavarga",
    label: GROUP_LABELS.kavarga,
    items: [
      { deva: "क", iast: "ka", group: "kavarga", articulation: "surda, não aspirada — da garganta", example: { word: "कर्मन्", iast: "karman" } },
      { deva: "ख", iast: "kha", group: "kavarga", articulation: "surda, aspirada — da garganta" },
      { deva: "ग", iast: "ga", group: "kavarga", articulation: "sonora, não aspirada — da garganta", example: { word: "गायत्री", iast: "gāyatrī" } },
      { deva: "घ", iast: "gha", group: "kavarga", articulation: "sonora, aspirada — da garganta" },
      { deva: "ङ", iast: "ṅa", group: "kavarga", articulation: "nasal — da garganta" },
    ],
  },
  {
    id: "cavarga",
    label: GROUP_LABELS.cavarga,
    items: [
      { deva: "च", iast: "ca", group: "cavarga", articulation: "surda, não aspirada — palatal", example: { word: "चन्द्रमा", iast: "candra" } },
      { deva: "छ", iast: "cha", group: "cavarga", articulation: "surda, aspirada — palatal" },
      { deva: "ज", iast: "ja", group: "cavarga", articulation: "sonora, não aspirada — palatal", example: { word: "ज्योतिष्", iast: "jyotiṣ" } },
      { deva: "झ", iast: "jha", group: "cavarga", articulation: "sonora, aspirada — palatal" },
      { deva: "ञ", iast: "ña", group: "cavarga", articulation: "nasal — palatal" },
    ],
  },
  {
    id: "tavarga",
    label: GROUP_LABELS.tavarga,
    items: [
      { deva: "ट", iast: "ṭa", group: "tavarga", articulation: "surda, não aspirada — retroflexa", example: { word: "पट", iast: "paṭa" } },
      { deva: "ठ", iast: "ṭha", group: "tavarga", articulation: "surda, aspirada — retroflexa" },
      { deva: "ड", iast: "ḍa", group: "tavarga", articulation: "sonora, não aspirada — retroflexa" },
      { deva: "ढ", iast: "ḍha", group: "tavarga", articulation: "sonora, aspirada — retroflexa" },
      { deva: "ण", iast: "ṇa", group: "tavarga", articulation: "nasal — retroflexa", example: { word: "प्राण", iast: "prāṇa" } },
    ],
  },
  {
    id: "tavarga2",
    label: GROUP_LABELS.tavarga2,
    items: [
      { deva: "त", iast: "ta", group: "tavarga2", articulation: "surda, não aspirada — dental", example: { word: "तत्", iast: "tat" } },
      { deva: "थ", iast: "tha", group: "tavarga2", articulation: "surda, aspirada — dental" },
      { deva: "द", iast: "da", group: "tavarga2", articulation: "sonora, não aspirada — dental", example: { word: "देव", iast: "deva" } },
      { deva: "ध", iast: "dha", group: "tavarga2", articulation: "sonora, aspirada — dental", example: { word: "धर्म", iast: "dharma" } },
      { deva: "न", iast: "na", group: "tavarga2", articulation: "nasal — dental", example: { word: "नर", iast: "nara" } },
    ],
  },
  {
    id: "pavarga",
    label: GROUP_LABELS.pavarga,
    items: [
      { deva: "प", iast: "pa", group: "pavarga", articulation: "surda, não aspirada — labial", example: { word: "पुरुष", iast: "puruṣa" } },
      { deva: "फ", iast: "pha", group: "pavarga", articulation: "surda, aspirada — labial" },
      { deva: "ब", iast: "ba", group: "pavarga", articulation: "sonora, não aspirada — labial", example: { word: "बन्धन", iast: "bandhana" } },
      { deva: "भ", iast: "bha", group: "pavarga", articulation: "sonora, aspirada — labial", example: { word: "भूमि", iast: "bhūmi" } },
      { deva: "म", iast: "ma", group: "pavarga", articulation: "nasal — labial", example: { word: "मनस्", iast: "manas" } },
    ],
  },
  {
    id: "antahstha",
    label: GROUP_LABELS.antahstha,
    items: [
      { deva: "य", iast: "ya", group: "antahstha", articulation: "semivogal — palatal", example: { word: "यज्ञ", iast: "yajña" } },
      { deva: "र", iast: "ra", group: "antahstha", articulation: "semivogal — retroflexa", example: { word: "रथ", iast: "ratha" } },
      { deva: "ल", iast: "la", group: "antahstha", articulation: "semivogal — dental", example: { word: "लोक", iast: "loka" } },
      { deva: "व", iast: "va", group: "antahstha", articulation: "semivogal — labial", example: { word: "विद्या", iast: "vidyā" } },
    ],
  },
  {
    id: "usman",
    label: GROUP_LABELS.usman,
    items: [
      { deva: "श", iast: "śa", group: "usman", articulation: "sibilante — palatal", example: { word: "शान्ति", iast: "śānti" } },
      { deva: "ष", iast: "ṣa", group: "usman", articulation: "sibilante — retroflexa", example: { word: "विष्णु", iast: "viṣṇu" } },
      { deva: "स", iast: "sa", group: "usman", articulation: "sibilante — dental", example: { word: "सत्य", iast: "satya" } },
      { deva: "ह", iast: "ha", group: "usman", articulation: "aspirada — da garganta", example: { word: "हिरण्य", iast: "hiraṇya" } },
    ],
  },
];

// ---------------------------------------------------------------- vocabulário

export const VOCAB: VocabEntry[] = [
  { deva: "अग्नि", iast: "agni", gloss: "fogo", meaning: "O fogo sacrificial, mediador entre humano e divino; primeira palavra do Ṛgveda.", query: "Agni fire sacrificial", category: "deva" },
  { deva: "इन्द्र", iast: "indra", gloss: "Indra", meaning: "Rei dos devas, guerreiro que liberta as águas com o vajra.", query: "Indra vajra waters", category: "deva" },
  { deva: "सूर्य", iast: "sūrya", gloss: "Sol", meaning: "O Sol, olho dos deuses e do mundo; atravessa o céu em seu carro.", query: "Surya sun eye gods", category: "deva" },
  { deva: "वायु", iast: "vāyu", gloss: "vento", meaning: "O vento, sopro e força vital que atravessa os mundos.", query: "Vayu wind breath worlds", category: "deva" },
  { deva: "उषस्", iast: "uṣas", gloss: "aurora", meaning: "A deusa Aurora, filha do céu, que desperta os seres e renova a luz.", query: "Ushas dawn goddess light", category: "deva" },
  { deva: "वाक्", iast: "vāk", gloss: "fala", meaning: "A Fala (Vāc), a Palavra primordial de que os ṛṣis extraíram os hinos.", query: "speech Vac word sages", category: "principio" },
  { deva: "यज्ञ", iast: "yajña", gloss: "sacrifício", meaning: "O sacrifício ritual — a ação que sustenta ṛta, a ordem cósmica.", query: "yajna sacrifice ritual order", category: "ritual" },
  { deva: "ऋत", iast: "ṛta", gloss: "ordem cósmica", meaning: "A ordem que rege o cosmos e o rito; raiz de dharma e de 'right' em inglês.", query: "rita cosmic order truth", category: "principio" },
  { deva: "सत्य", iast: "satya", gloss: "verdade", meaning: "A verdade-real; no Ṛgveda, um dos nomes do caminho do ṛta.", query: "satya truth real", category: "principio" },
  { deva: "धर्म", iast: "dharma", gloss: "lei, dever", meaning: "O sustentador: lei cósmica e dever — da raiz dhṛ, 'sustentar'.", query: "dharma law duty sustain", category: "principio" },
  { deva: "कर्मन्", iast: "karman", gloss: "ação", meaning: "Ação ritual e ação em geral, com suas consequências.", query: "karma action deed", category: "principio" },
  { deva: "ब्रह्मन्", iast: "brahman", gloss: "o Absoluto", meaning: "O Absoluto nos Upaniṣads: realidade única por trás de todos os fenômenos.", query: "brahman absolute reality one", category: "principio" },
  { deva: "आत्मन्", iast: "ātman", gloss: "o Si-mesmo", meaning: "O Si-mesmo que os Upaniṣads identificam com brahman (tat tvam asi).", query: "atman self brahman", category: "principio" },
  { deva: "ओम्", iast: "oṃ", gloss: "o sílabo OM", meaning: "O sílabo sagrado — passado, presente e futuro, e o que além deles (Māṇḍūkya).", query: "Om syllable sacred past future", category: "principio" },
  { deva: "पूर्ण", iast: "pūrṇa", gloss: "pleno", meaning: "Plenitude: pūrṇam adaḥ pūrṇam idam — aquilo é pleno, isto é pleno (Īśā).", query: "purnam fullness complete that this", category: "principio" },
  { deva: "मन्त्र", iast: "mantra", gloss: "mantra", meaning: "Instrumento do pensamento (man-tra): fórmula sagrada que realiza o que nomeia.", query: "mantra formula sacred thought", category: "ritual" },
  { deva: "ऋच्", iast: "ṛc", gloss: "hino, verso", meaning: "Um verso do Ṛgveda; da raiz arc, 'celebrar'.", query: "ric verse hymn celebrate", category: "ritual" },
  { deva: "छन्दस्", iast: "chandas", gloss: "metro védico", meaning: "Metro (gāyatrī, triṣṭubh…) e, depois, 'texto revelado'.", query: "chandas meter gayatri verse", category: "ritual" },
  { deva: "ऋषि", iast: "ṛṣi", gloss: "vidente", meaning: "O vidente que 'viu' o hino — autor tradicional de cada ṛc.", query: "rishi seer vision hymn", category: "ritual" },
  { deva: "गायत्री", iast: "gāyatrī", gloss: "metro gāyatrī", meaning: "Metro de 24 sílabas; também a fórmula solar RV 3.62.10.", query: "gayatri meter solar formula", category: "ritual" },
  { deva: "सोम", iast: "soma", gloss: "Soma", meaning: "A planta/elixir pressado no rito; rei das plantas, inteiro hino IX.", query: "soma plant pressed juice", category: "ritual" },
  { deva: "हविस्", iast: "havis", gloss: "oblação", meaning: "A oblação derramada no fogo sacrificial.", query: "oblation offering poured fire", category: "ritual" },
  { deva: "द्यौस्", iast: "dyauḥ", gloss: "céu", meaning: "O Céu diurno, pai — par de pṛthivī, a Terra (dyāvāpṛthivī).", query: "dyaus sky father heaven", category: "cosmos" },
  { deva: "पृथिवी", iast: "pṛthivī", gloss: "Terra", meaning: "A Terra, a larga, mãe — par do céu dyauḥ.", query: "prithivi earth broad mother", category: "cosmos" },
  { deva: "अन्तरिक्ष", iast: "antarikṣa", gloss: "espaço intermediário", meaning: "O espaço entre céu e terra — o ar onde voam os pássaros sagrados.", query: "antariksha intermediate space air", category: "cosmos" },
  { deva: "लोक", iast: "loka", gloss: "mundo", meaning: "Mundo/esfera — os três lokas e o loka do luzente.", query: "loka world sphere bright", category: "cosmos" },
  { deva: "अमृत", iast: "amṛta", gloss: "imortalidade", meaning: "O que não morre — néctar da imortalidade, oposto de mṛtyu.", query: "amrita immortality nectar death", category: "cosmos" },
  { deva: "नासदीय", iast: "nāsadīya", gloss: "'não era'", meaning: "Nome do hino da criação RV 10.129: 'então não havia nem o ser nem o não-ser'.", query: "Nasadiya creation hymn neither being", category: "cosmos" },
  { deva: "पुरुष", iast: "puruṣa", gloss: "Pessoa cósmica", meaning: "A Pessoa que os deuses dispõem como oferenda — mil cabeças, mil olhos (RV 10.90).", query: "Purusha cosmic person thousand eyes", category: "cosmos" },
  { deva: "हिरण्यगर्भ", iast: "hiraṇyagarbha", gloss: "embrião dourado", meaning: "O embrião dourado: origem do mundo (RV 10.121).", query: "Hiranyagarbha golden embryo arose", category: "cosmos" },
];

export const VOCAB_CATEGORIES: { id: VocabEntry["category"]; label: string }[] = [
  { id: "deva", label: "Devas" },
  { id: "ritual", label: "Ritual e palavra" },
  { id: "principio", label: "Princípios" },
  { id: "cosmos", label: "Cosmos" },
];

// ---------------------------------------------------------------- lições

export const LESSONS: Lesson[] = [
  {
    id: "sandhi",
    title: "Sandhi — a fusão dos sons",
    subtitle: "Por que 'agnim īḷe' não é 'agni īḷe': as regras que moldam o verso",
    sections: [
      {
        heading: "O que é sandhi",
        body: [
          "No sânscrito, os sons não toleram encontros: vogais e palavras fundem-se nas juntas (sandhi = 'amarração'). O que na prosa aparece separado, no verso vem unido pela eufonia.",
          "Regras frequentes no corpus: a + i → e; ā + i → e; a + u → o; ā + u → o; i + i → ī; u + u → ū; e + a → e. Finais -m tornam-se anusvāra antes de consoantes.",
        ],
        examples: [
          { deva: "अग्निमीळे", iast: "agnim īḷe", gloss: "'louvo Agni' — RV 1.1.1: a junção abertura do Ṛgveda" },
          { deva: "पूर्णमदः", iast: "pūrṇam adaḥ", gloss: "'aquilo é pleno' — Īśā: pūrṇam + adaḥ" },
          { deva: "तत्त्वमसि", iast: "tat tvam asi", gloss: "'isso és tu' — Chāndogya 6: tat + tvam funde-se em tt" },
        ],
      },
      {
        heading: "Visarga e anusvāra",
        body: [
          "O visarga (ḥ) — sopro final aspirado — muda conforme o que vem depois: antes de k/kh → k, antes de c/ch → ś, antes de t/th → s, antes de p/ph → p, antes de vogais sonoras → r. É por isso que 'viṣṇuḥ + u' vira 'viṣṇor'... na junção.",
          "O anusvāra (ṃ) é nasalização: substitui -m final antes de consoante. 'oṃ' é o sílabo OM — a nasalização sustentada que fecha cada recitação.",
        ],
        examples: [
          { deva: "विष्णोर्नु कं", iast: "viṣṇor nu kaṃ", gloss: "viṣṇuḥ + u + kam: visarga + u → o(r)" },
          { deva: "ओम् शान्तिः", iast: "oṃ śāntiḥ", gloss: "OM, paz — o fecho tradicional dos Upaniṣads" },
        ],
      },
    ],
  },
  {
    id: "nominais",
    title: "Subantas — os nomes em caso",
    subtitle: "Oito casos: quem age, a quem se oferenta, com o que se sacrifica",
    sections: [
      {
        heading: "O sistema de casos",
        body: [
          "O substantivo (subanta) declina-se em oito casos: nominativo (sujeito), acusativo (objeto), instrumental (meio), dativo (destinatário), ablativo (origem), genitivo (posse), locativo (espaço/tempo), vocativo (chamado).",
          "No hino a Agni (RV 1.1), os casos estruturam o rito: agnim īḷe (acusativo — 'louvo Agni'), hotāram (acusativo — 'o sacerdote'), viśvāya karmaṇe — dativo duplo, 'para o ato de todos'.",
        ],
        examples: [
          { deva: "अग्निः", iast: "agniḥ", gloss: "nominativo — Agni (age)" },
          { deva: "अग्निम्", iast: "agnim", gloss: "acusativo — a Agni (recebe a ação)" },
          { deva: "अग्नये", iast: "agnaye", gloss: "dativo — para Agni (destinatário)" },
          { deva: "अग्निना", iast: "agninā", gloss: "instrumental — por/com Agni (meio)" },
          { deva: "अग्नौ", iast: "agnau", gloss: "locativo — em Agni (no fogo)" },
        ],
      },
    ],
  },
  {
    id: "verbos",
    title: "Tiṅantas — o verbo no rito",
    subtitle: "Presente, imperativo e imperfeito: a voz do sacrifício",
    sections: [
      {
        heading: "Formas centrais",
        body: [
          "O verbo (tiṅanta) conjuga-se em pessoa (1/2/3), número (singular/dual/plural) e tempo-modo. No Ṛgveda dominam o presente ('īḷe' — eu louvo), o imperativo de convite ('agāhi' — vem!) e o subjuntivo de propósito.",
          "Duas vozes: parasmaipada (ação 'para o outro') e ātmanepada ('para si'). īḷe é ātmanepada — o louvor retorna ao próprio vidente.",
        ],
        examples: [
          { deva: "ईळे", iast: "īḷe", gloss: "presente ātmanepada 1ª sing. — 'eu louvo'" },
          { deva: "पातु", iast: "pātu", gloss: "imperativo 3ª sing. — 'que ele proteja'" },
          { deva: "अगाहि", iast: "agāhi", gloss: "imperativo 2ª sing. — 'vem!' (convite ao fogo)" },
          { deva: "अभवत्", iast: "abhavat", gloss: "imperfeito 3ª sing. — 'foi/veio a ser'" },
        ],
      },
    ],
  },
];

// ---------------------------------------------------------------- fonética e voz (Web Speech API)

/**
 * Converte devanāgarī para aproximação fonética legível por qualquer voz sintetizada
 * (inglês, português ou padrão do SO), caso o dispositivo não possua voz nativa em Hindi/Sânscrito.
 * Isso impede que navegadores no macOS/Windows fiquem em silêncio absoluto ao tentar ler glifos devanāgarī.
 */
export function devaToPhonetic(text: string): string {
  if (!/[\u0900-\u097F]/.test(text)) {
    return text;
  }

  const vowels: Record<string, string> = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ऋ": "ri", "ॠ": "ree", "ऌ": "li", "ॡ": "lee",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
  };

  const matras: Record<string, string> = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo",
    "ृ": "ri", "ॄ": "ree", "ॢ": "li", "ॣ": "lee",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
  };

  const consonants: Record<string, string> = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "ळ": "l", "क्ष": "ksha", "ज्ञ": "gya",
  };

  let out = "";
  let i = 0;
  while (i < text.length) {
    const ch = text[i];
    if (ch === "ॐ") {
      out += " Om ";
      i++;
      continue;
    }
    if (vowels[ch]) {
      out += vowels[ch];
      i++;
      continue;
    }
    if (consonants[ch]) {
      const cons = consonants[ch];
      const next = text[i + 1];
      if (next === "्") {
        out += cons;
        i += 2;
      } else if (matras[next]) {
        out += cons + matras[next];
        i += 2;
      } else {
        out += cons + "a";
        i++;
      }
      continue;
    }
    if (ch === "ं") {
      out += "m";
      i++;
      continue;
    }
    if (ch === "ः") {
      out += "h";
      i++;
      continue;
    }
    if (ch === "ऽ") {
      out += "'";
      i++;
      continue;
    }
    if (ch === "्") {
      i++;
      continue;
    }
    // Acentos védicos / danda
    if (ch >= "\u0951" && ch <= "\u0954") {
      i++;
      continue;
    }
    if (ch === "।" || ch === "॥") {
      out += ". ";
      i++;
      continue;
    }
    out += ch;
    i++;
  }
  return out.replace(/\s+/g, " ").trim();
}

const _CONS = [
  "kṣ", "jñ",
  "kh", "gh", "ch", "jh", "ṭh", "ḍh", "th", "dh", "ph", "bh",
  "ṅ", "ñ", "ṭ", "ḍ", "ṇ", "ś", "ṣ",
  "k", "g", "c", "j", "t", "d", "n", "p", "b", "m", "y", "r", "l", "v", "s", "h",
];
const _VOWELS = ["ai", "au", "ā", "ī", "ū", "ṝ", "ḹ", "ṛ", "e", "o", "a", "i", "u"];

const _BASE: Record<string, string> = {
  k: "k", kh: "k", g: "g", gh: "g", ṅ: "ng",
  c: "ch", ch: "ch", j: "j", jh: "j", ñ: "ny",
  ṭ: "t", ṭh: "t", ḍ: "d", ḍh: "d", ṇ: "n",
  t: "t", th: "t", d: "d", dh: "d", n: "n",
  p: "p", ph: "p", b: "b", bh: "b", m: "m",
  y: "y", r: "r", l: "l", v: "v",
  ś: "sh", ṣ: "shr", s: "s", h: "h",
  kṣ: "ksh", jñ: "gny",
};
const _VOWEL_SOUND: Record<string, string> = {
  a: "uh", ā: "aah", i: "ih", ī: "ee", u: "oo", ū: "ooo",
  ṛ: "rih", ṝ: "ree", ḷ: "lih", ḹ: "lee",
  e: "eh", ai: "eye", o: "oh", au: "ow",
};

function _aspirated(cons: string): boolean {
  return cons.length > 1 && cons.endsWith("h") && cons !== "h";
}

function _withVowel(cons: string, vowel: string): string {
  const head = (_BASE[cons] ?? cons) + _VOWEL_SOUND[vowel];
  return _aspirated(cons) ? `${head}-huh` : head;
}

function _starts(s: string, i: number, tokens: string[]): string | null {
  for (const t of tokens) {
    if (s.startsWith(t, i)) return t;
  }
  return null;
}

/**
 * Reescreve IAST para uma voz latina.
 * "a" sozinho seria lido como o nome da letra ("ei"); "tha"/"pha" virariam
 * o th e o f do inglês. Cada sílaba sai como "kuh", "kuh-huh", "tuh-huh".
 */
export function iastToSpeakable(iast: string): string {
  const src = (iast || "")
    .replace(/ṁ/g, "ṃ")
    .replace(/[|।॥]/g, " ")
    .trim();
  if (!src) return "";
  if (/[\u0900-\u097F]/.test(src) && !/[a-zA-Zāīūṛṝḷḹṃḥṅñṭḍṇśṣ]/.test(src)) {
    return iastToSpeakable(devaToIast(src));
  }

  const syllables: string[] = [];
  let pending: string[] = [];
  let i = 0;
  const flushBare = () => {
    if (!pending.length) return;
    const bare = pending.map((c) => _BASE[c] ?? c).join("");
    pending = [];
    if (syllables.length) syllables[syllables.length - 1] += bare;
    else syllables.push(bare);
  };

  while (i < src.length) {
    const ch = src[i];
    if (ch === " " || ch === "\n" || ch === "\t") {
      flushBare();
      i++;
      continue;
    }
    if (ch === "ṃ" || ch === "ḥ" || ch === "m̐") {
      flushBare();
      const mark = ch === "ḥ" ? "-huh" : "m";
      if (syllables.length && syllables[syllables.length - 1] !== " ") syllables[syllables.length - 1] += mark;
      else syllables.push(ch === "ḥ" ? "huh" : "m");
      i++;
      continue;
    }
    const vowel = _starts(src, i, _VOWELS);
    if (vowel) {
      const onset = pending;
      pending = [];
      const head = onset.slice(0, -1).map((c) => _BASE[c] ?? c).join("");
      const nucleus = onset.length
        ? head + _withVowel(onset[onset.length - 1], vowel)
        : _VOWEL_SOUND[vowel];
      syllables.push(nucleus);
      i += vowel.length;
      continue;
    }
    // ळ/ḷ antes de vogal é a consoante retroflexa (ईळे = īḷe), não a vogal ऌ.
    if (src.startsWith("ḷ", i) && _starts(src, i + "ḷ".length, _VOWELS)) {
      pending.push("l");
      i += "ḷ".length;
      continue;
    }
    if (src.startsWith("ḷ", i)) {
      flushBare();
      syllables.push(_VOWEL_SOUND["ḷ"]);
      i += "ḷ".length;
      continue;
    }
    const cons = _starts(src, i, _CONS);
    if (cons) {
      pending.push(cons);
      i += cons.length;
      continue;
    }
    flushBare();
    i++;
  }
  flushBare();
  return syllables.filter((s) => s !== " ").join(" ").replace(/\s+/g, " ").trim();
}

/** Devanāgarī → IAST, o bastante para a pronúncia (vogal inerente incluída). */
export function devaToIast(text: string): string {
  const vowels: Record<string, string> = {
    "अ": "a", "आ": "ā", "इ": "i", "ई": "ī", "उ": "u", "ऊ": "ū",
    "ऋ": "ṛ", "ॠ": "ṝ", "ऌ": "ḷ", "ॡ": "ḹ",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
  };
  const matras: Record<string, string> = {
    "ा": "ā", "ि": "i", "ी": "ī", "ु": "u", "ू": "ū",
    "ृ": "ṛ", "ॄ": "ṝ", "ॢ": "ḷ", "ॣ": "ḹ",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
  };
  const consonants: Record<string, string> = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ṅ",
    "च": "c", "छ": "ch", "ज": "j", "झ": "jh", "ञ": "ñ",
    "ट": "ṭ", "ठ": "ṭh", "ड": "ḍ", "ढ": "ḍh", "ण": "ṇ",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "ś", "ष": "ṣ", "स": "s", "ह": "h",
    "ळ": "ḷ",
  };
  let out = "";
  let i = 0;
  while (i < text.length) {
    const ch = text[i];
    if (ch === "ॐ") {
      out += "oṃ";
      i++;
      continue;
    }
    if (vowels[ch]) {
      out += vowels[ch];
      i++;
      continue;
    }
    if (consonants[ch]) {
      const next = text[i + 1];
      if (next === "्") {
        out += consonants[ch];
        i += 2;
      } else if (matras[next]) {
        out += consonants[ch] + matras[next];
        i += 2;
      } else {
        out += consonants[ch] + "a";
        i++;
      }
      continue;
    }
    if (ch === "ं" || ch === "ँ") {
      out += "ṃ";
      i++;
      continue;
    }
    if (ch === "ः") {
      out += "ḥ";
      i++;
      continue;
    }
    if (ch === "ऽ" || (ch >= "\u0951" && ch <= "\u0954")) {
      i++;
      continue;
    }
    if (ch === "।" || ch === "॥") {
      out += " ";
      i++;
      continue;
    }
    if (ch !== "्") out += ch;
    i++;
  }
  return out.replace(/\s+/g, " ").trim();
}

let cachedVoices: SpeechSynthesisVoice[] = [];

function getVoicesList(): SpeechSynthesisVoice[] {
  if (typeof window === "undefined" || !window.speechSynthesis) return [];
  const list = window.speechSynthesis.getVoices();
  if (list && list.length > 0) {
    cachedVoices = list;
  }
  return cachedVoices.length > 0 ? cachedVoices : list;
}

if (typeof window !== "undefined" && window.speechSynthesis) {
  try {
    getVoicesList();
    window.speechSynthesis.addEventListener("voiceschanged", () => {
      getVoicesList();
    });
  } catch {
    /* ignore */
  }
}

function normLang(lang: string): string {
  return lang.replace("_", "-").toLowerCase();
}

function isIndicLang(lang: string): boolean {
  const n = normLang(lang);
  return n.startsWith("hi") || n.startsWith("sa");
}

/**
 * Seleciona voz hi/sa. Sem ela, devolve null e a leitura usa a soletração latina.
 */
export function pickSanskritVoice(): SpeechSynthesisVoice | null {
  const list = getVoicesList();
  if (!list.length) return null;
  const norm = (v: SpeechSynthesisVoice) => normLang(v.lang);

  // Hindi/sânscrito em qualquer serviço: uma voz inglesa local leria "a" como "ei".
  const indicList = list.filter(
    (v) => isIndicLang(v.lang) || /lekha|sanskrit|hindi/i.test(v.name)
  );
  const pool = indicList.length > 0 ? indicList : list.filter((v) => v.localService !== false);
  const voices = pool.length > 0 ? pool : list;

  for (const want of ["sa-in", "hi-in", "sa", "hi"]) {
    const exact = pool.find((v) => norm(v) === want);
    if (exact) return exact;
  }
  const byName = pool.find(
    (v) =>
      v.name.toLowerCase().includes("lekha") ||
      v.name.toLowerCase().includes("sanya") ||
      v.name.toLowerCase().includes("hindi") ||
      v.name.toLowerCase().includes("sanskrit")
  );
  if (byName) return byName;
  const indic = voices.find((v) => norm(v).startsWith("hi") || norm(v).startsWith("sa"));
  if (indic) return indic;
  return null;
}

/** Voz capaz de ler IAST/fonética: inglês primeiro, depois português. */
export function pickLatinVoice(): SpeechSynthesisVoice | null {
  const list = getVoicesList();
  if (!list.length) return null;
  const norm = (v: SpeechSynthesisVoice) => normLang(v.lang);
  const localList = list.filter((v) => v.localService !== false);
  const pool = localList.length > 0 ? localList : list;

  for (const want of ["en-us", "en-gb", "en-in", "en", "pt-br", "pt-pt", "pt"]) {
    const exact = pool.find((v) => norm(v) === want);
    if (exact) return exact;
  }
  return (
    pool.find((v) => norm(v).startsWith("en")) ||
    pool.find((v) => norm(v).startsWith("pt")) ||
    pool[0] ||
    null
  );
}

export function pickPortugueseVoice(): SpeechSynthesisVoice | null {
  const list = getVoicesList();
  if (!list.length) return null;
  const norm = (v: SpeechSynthesisVoice) => normLang(v.lang);
  const localList = list.filter((v) => v.localService !== false);
  const pool = localList.length > 0 ? localList : list;

  for (const want of ["pt-br", "pt-pt", "pt"]) {
    const exact = pool.find((v) => norm(v) === want);
    if (exact) return exact;
  }
  const byName = pool.find(
    (v) =>
      v.name.toLowerCase().includes("luciana") ||
      v.name.toLowerCase().includes("eddy") ||
      v.name.toLowerCase().includes("portuguese") ||
      v.name.toLowerCase().includes("português")
  );
  if (byName) return byName;
  return pool.find((v) => norm(v).startsWith("pt")) || null;
}

// Conjunto para evitar que o Garbage Collector do V8/WebKit descarte o Utterance antes do término
const activeUtterances = new Set<SpeechSynthesisUtterance>();

export function stopSpeaking(): void {
  if (typeof window !== "undefined" && window.speechSynthesis) {
    try {
      window.speechSynthesis.cancel();
      activeUtterances.clear();
      if (window.speechSynthesis.paused) {
        window.speechSynthesis.resume();
      }
    } catch {
      /* ignore */
    }
  }
}

/**
 * Dispara a síntese de fala imediatamente no mesmo ciclo de eventos do clique do usuário.
 * IMPORTANTE: Nunca usar setTimeout antes de synth.speak(), pois navegadores modernos (Safari/Chrome)
 * revogam o gesto do usuário ("user activation") em callbacks assíncronos e descartam o áudio.
 */
function safeSpeak(utter: SpeechSynthesisUtterance, onEnd?: () => void): void {
  if (typeof window === "undefined" || !window.speechSynthesis) {
    onEnd?.();
    return;
  }
  const synth = window.speechSynthesis;

  activeUtterances.add(utter);

  let finished = false;
  let safetyTimer: ReturnType<typeof setTimeout> | null = null;

  const cleanup = () => {
    if (!finished) {
      finished = true;
      if (safetyTimer) {
        clearTimeout(safetyTimer);
        safetyTimer = null;
      }
      activeUtterances.delete(utter);
      onEnd?.();
    }
  };

  utter.onend = () => {
    cleanup();
  };
  utter.onerror = () => {
    cleanup();
  };

  // Timer de segurança longo para não travar a UI caso o motor do browser não dispare onend
  safetyTimer = setTimeout(cleanup, Math.max(6000, utter.text.length * 250));

  try {
    synth.cancel();
    if (synth.paused) {
      synth.resume();
    }
    synth.speak(utter);
  } catch {
    cleanup();
  }
}

export type SanskritSpeechPlan = {
  text: string;
  lang: string;
  voice: SpeechSynthesisVoice | null;
};

/**
 * Decide o que a síntese vai ler.
 * Voz indiana lê o devanāgarī. As outras recebem uma soletração ("kuh",
 * "kuh-huh"), nunca o IAST cru — "a" e "tha" sairiam como letras inglesas.
 */
export function planSanskritSpeech(
  text: string,
  iastFallback?: string,
  voice: SpeechSynthesisVoice | null = null,
  latinVoice: SpeechSynthesisVoice | null = null,
): SanskritSpeechPlan {
  if (voice && isIndicLang(voice.lang)) {
    return { text, lang: voice.lang, voice };
  }
  const source = (iastFallback && iastFallback.trim()) || text;
  const spoken = iastToSpeakable(source);
  const english = voice && normLang(voice.lang).startsWith("en") ? voice : null;
  const reader = english || latinVoice;
  return { text: spoken, lang: reader?.lang || "en-US", voice: reader };
}

export function speakSanskrit(text: string, iastFallback?: string, onEnd?: () => void, rate = 0.85): void {
  if (typeof window === "undefined" || !window.speechSynthesis) {
    onEnd?.();
    return;
  }

  const plan = planSanskritSpeech(text, iastFallback, pickSanskritVoice(), pickLatinVoice());
  const utter = new SpeechSynthesisUtterance(plan.text);
  if (plan.voice) {
    utter.voice = plan.voice;
  }
  utter.lang = plan.lang;
  utter.rate = rate;
  safeSpeak(utter, onEnd);
}

export function speakSanskritWithPromise(text: string, iastFallback?: string, rate = 0.85): Promise<void> {
  return new Promise((resolve) => {
    speakSanskrit(text, iastFallback, resolve, rate);
  });
}

export function speakPortuguese(text: string, onEnd?: () => void): void {
  if (typeof window === "undefined" || !window.speechSynthesis) {
    onEnd?.();
    return;
  }
  const utter = new SpeechSynthesisUtterance(text);
  const voice = pickPortugueseVoice();
  if (voice) {
    utter.voice = voice;
    utter.lang = voice.lang;
  } else {
    utter.lang = "pt-BR";
  }
  utter.rate = 0.95;
  safeSpeak(utter, onEnd);
}