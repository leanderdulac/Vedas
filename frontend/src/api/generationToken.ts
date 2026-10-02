/**
 * Bearer opcional de geração (dev local). Em produção não coloque o token
 * no bundle — use proxy de autenticação ou provider=extractive.
 */
export function generationAuthHeaders(token?: string): Record<string, string> {
  const raw =
    token !== undefined ? token : String(import.meta.env.VITE_GENERATION_API_TOKEN ?? "");
  const trimmed = raw.trim();
  return trimmed ? { Authorization: `Bearer ${trimmed}` } : {};
}
