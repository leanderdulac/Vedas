// Modo eco de recitação: o app recita um pāda, pausa, o usuário repete.
// Versos curados do corpus (verificados no índice) + ritmo da pausa de eco.

export type RecitationEntry = {
  verseId: string;
  label: string;
  work: string;
  note: string;
};

export const RECITATIONS: RecitationEntry[] = [
  {
    verseId: "RV.1.1.1",
    label: "RV 1.1.1 — a Agni",
    work: "Ṛgveda",
    note: "O hino inicial: 'Agni louvo, o sacerdote da casa, o deva do sacrifício'.",
  },
  {
    verseId: "RV.10.129.1",
    label: "RV 10.129.1 — Nāsadīya",
    work: "Ṛgveda",
    note: "'Não era o ser, nem o não-ser' — o hino da criação.",
  },
  {
    verseId: "RV.10.90.1",
    label: "RV 10.90.1 — Puruṣa",
    work: "Ṛgveda",
    note: "A Pessoa cósmica de mil cabeças, mil olhos, mil pés.",
  },
  {
    verseId: "BG.4.7",
    label: "BG 4.7 — yadā yadā hi dharmasya",
    work: "Bhagavadgītā",
    note: "'Sempre que o dharma declina, ó Bhārata' — a promessa do avatar.",
  },
  {
    verseId: "AV.1.1.1",
    label: "AV 1.1.1 — a Vācaspati",
    work: "Atharvaveda",
    note: "'Vācaspati, força dos que carregam todas as formas — conceda-me hoje vigor'.",
  },
  {
    verseId: "VS.1.1",
    label: "VS 1.1 — iṣe tvā",
    work: "Vājasaneyi-saṃhitā",
    note: "A fórmula de oblação que abre o Yajurveda.",
  },
];

type PadaLike = { sa?: string; iast?: string } | null | undefined;

/** Pausa de eco: o usuário repite o pāda antes do app seguir adiante. */
export function echoGapMs(pada: PadaLike): number {
  const text = (pada?.sa || pada?.iast || "").trim();
  if (!text) return 3200;
  return Math.min(9000, Math.max(3200, 2400 + text.length * 240));
}