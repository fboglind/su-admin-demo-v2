/**
 * UO (Utbildningsområde) constants and helpers.
 * Shared across all frontend components.
 */

export const UO_META = {
  2434: { abbr: "HU", name: "Humaniora", nameEn: "Humanities", color: "#8B5CF6" },
  2436: { abbr: "JU", name: "Juridik", nameEn: "Law", color: "#EC4899" },
  2438: { abbr: "LU", name: "Lärarutbildning", nameEn: "Teacher Education", color: "#F59E0B" },
  2439: { abbr: "ME", name: "Medicin", nameEn: "Medicine", color: "#EF4444" },
  2441: { abbr: "NA", name: "Naturvetenskap", nameEn: "Natural Science", color: "#10B981" },
  2442: { abbr: "SA", name: "Samhällsvetenskap", nameEn: "Social Sciences", color: "#3B82F6" },
  2444: { abbr: "TE", name: "Teknik", nameEn: "Technology", color: "#6366F1" },
  2445: { abbr: "VÅ", name: "Vård", nameEn: "Health Care", color: "#14B8A6" },
  2447: { abbr: "ÖV", name: "Övrigt", nameEn: "Other", color: "#78716C" },
  2451: { abbr: "VU", name: "Verksamhetsförlagd utb.", nameEn: "Work-based Ed.", color: "#D946EF" },
};

export const UO_CODES = Object.keys(UO_META).map(Number).sort();

export const getUO = (code) => UO_META[code] || { abbr: `?${code}`, name: "Unknown", color: "#999" };

/**
 * Convert predictions object { 2434: 0.8, 2441: 0.2, ... } to sorted array
 * for chart rendering.
 */
export const predsToChartData = (preds, type = "binary") => {
  return UO_CODES.map((code) => ({
    code,
    abbr: UO_META[code].abbr,
    name: UO_META[code].name,
    color: UO_META[code].color,
    value: preds?.[code] ?? 0,
  }));
};

/**
 * Parse gold label strings like "(2434, 2441)" into arrays.
 */
export const parseLabels = (labelStr) => {
  if (!labelStr || labelStr === "nan") return [];
  const matches = labelStr.match(/\d{4}/g);
  return matches ? matches.map(Number) : [];
};

export const API_BASE = import.meta.env.VITE_API_URL || "";

/**
 * Fetch helper with error handling.
 */
export const apiFetch = async (path, options = {}) => {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || `API error: ${res.status}`);
  }
  return res.json();
};
