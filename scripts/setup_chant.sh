#!/usr/bin/env bash
# Instala o canto local (VEDIC_RECITATION_STYLE=chant) fora do repositório:
#   - ACE-Step 1.5 (MIT) num checkout próprio com venv uv (Python 3.12, MPS/MLX no Mac)
#   - Demucs 4.1 (MIT) no mesmo venv, para isolar a voz
#   - checkpoints do ACE-Step (~9,4 GB: turbo DiT, LM 1.7B, VAE, Qwen3-Embedding)
#   - tanpura de sankalp (Freesound #155272, CC BY 4.0) em data/assets/ (fora do git)
# Nada disso entra no git. Depois: VEDIC_RECITATION_STYLE=chant e VEDIC_ACESTEP_DIR no .env.
#
#   scripts/setup_chant.sh [destino]   (padrão: ~/.local/share/vedas/ACE-Step-1.5)
set -euo pipefail
DEST="${1:-$HOME/.local/share/vedas/ACE-Step-1.5}"
REPO_URL="https://github.com/ace-step/ACE-Step-1.5.git"
COMMIT="ca1e85fe9430179831e6bc6be790c332190a3866"  # versão validada em 2026-10-04
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

command -v uv >/dev/null || { echo "instale o uv (https://docs.astral.sh/uv/)"; exit 1; }
command -v ffmpeg >/dev/null || { echo "instale o ffmpeg"; exit 1; }

if [ ! -d "$DEST/.git" ]; then
  git clone "$REPO_URL" "$DEST"
fi
git -C "$DEST" fetch --quiet origin && git -C "$DEST" checkout --quiet "$COMMIT"
(cd "$DEST" && uv sync)
TORCH="$("$DEST/.venv/bin/python" -c 'import torch; print(torch.__version__.split("+")[0])')"
TA="$("$DEST/.venv/bin/python" -c 'import torchaudio; print(torchaudio.__version__.split("+")[0])')"
printf 'torch==%s\ntorchaudio==%s\n' "$TORCH" "$TA" > "$DEST/.vedas-constraints.txt"
uv pip install --python "$DEST/.venv/bin/python" -c "$DEST/.vedas-constraints.txt" "demucs==4.1.0"
(cd "$DEST" && .venv/bin/python -c "from acestep.model_downloader import download_main_model as d; print(d(prefer_source='huggingface'))")
"$DEST/.venv/bin/python" -c "from demucs.pretrained import get_model; get_model('htdemucs')"
(cd "$HERE" && python3 scripts/fetch_tanpura.py)
echo
echo "Pronto. No .env do Vedas:"
echo "  VEDIC_RECITATION_STYLE=chant"
echo "  VEDIC_ACESTEP_DIR=$DEST"
