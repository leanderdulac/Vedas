/**
 * Helpers puros do token do pipeline (sem import.meta.env) para permitir
 * testes em Node. O token vive em sessionStorage — nunca em VITE_* nem no bundle.
 */
export const PIPELINE_TOKEN_KEY = "vedas_pipeline_token";

export function getStoredToken(
  storage?: Pick<Storage, "getItem"> | null
): string {
  if (!storage) return "";
  return storage.getItem(PIPELINE_TOKEN_KEY) ?? "";
}

export function storeToken(
  token: string,
  storage?: Pick<Storage, "setItem" | "removeItem"> | null
): void {
  if (!storage) return;
  if (token.trim()) storage.setItem(PIPELINE_TOKEN_KEY, token.trim());
  else storage.removeItem(PIPELINE_TOKEN_KEY);
}
