import { useState, useEffect } from "react";
import UODistributionChart from "../components/UODistributionChart";
import { apiFetch, UO_META, UO_CODES, getUO, parseLabels } from "../utils/uo";

export default function Compare() {
  const [courses, setCourses] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [disagreementsOnly, setDisagreementsOnly] = useState(false);
  const [uoFilter, setUoFilter] = useState("");
  const [expandedIdx, setExpandedIdx] = useState(null);

  const perPage = 15;

  useEffect(() => {
    loadData();
  }, [page, disagreementsOnly, uoFilter]);

  const loadData = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        page,
        per_page: perPage,
        disagreements: disagreementsOnly,
      });
      if (uoFilter) params.set("uo", uoFilter);

      const data = await apiFetch(`/api/compare?${params}`);
      setCourses(data.courses);
      setTotal(data.total);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const totalPages = Math.ceil(total / perPage);

  const getLabelChips = (labelsStr) => {
    const codes = parseLabels(labelsStr);
    return codes.map((code) => {
      const uo = getUO(code);
      return (
        <span
          key={code}
          className="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium text-white"
          style={{ backgroundColor: uo.color }}
        >
          {uo.abbr}
        </span>
      );
    });
  };

  const getPredChips = (predsObj) => {
    if (!predsObj) return <span className="text-gray-400 text-xs">—</span>;
    const active = Object.entries(predsObj)
      .filter(([_, v]) => v >= 0.5)
      .map(([code]) => Number(code));
    if (active.length === 0) return <span className="text-gray-400 text-xs">none</span>;
    return active.map((code) => {
      const uo = getUO(code);
      return (
        <span
          key={code}
          className="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium border"
          style={{ borderColor: uo.color, color: uo.color }}
        >
          {uo.abbr}
        </span>
      );
    });
  };

  const checkMatch = (goldStr, predsObj) => {
    if (!predsObj) return null;
    const gold = new Set(parseLabels(goldStr));
    const pred = new Set(
      Object.entries(predsObj)
        .filter(([_, v]) => v >= 0.5)
        .map(([code]) => Number(code))
    );
    if (gold.size === 0) return null;
    const same = gold.size === pred.size && [...gold].every((c) => pred.has(c));
    return same;
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Model Comparison</h1>
        <p className="text-gray-500 mt-1">
          Compare TF-IDF baseline vs BERT predictions on the validation set ({total} courses).
        </p>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-4 mb-6">
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={disagreementsOnly}
            onChange={(e) => {
              setDisagreementsOnly(e.target.checked);
              setPage(1);
            }}
            className="rounded border-gray-300"
          />
          <span className="text-gray-700">Show disagreements only</span>
        </label>

        <select
          value={uoFilter}
          onChange={(e) => {
            setUoFilter(e.target.value);
            setPage(1);
          }}
          className="border border-gray-300 rounded-md px-3 py-1.5 text-sm"
        >
          <option value="">All UO categories</option>
          {UO_CODES.map((code) => (
            <option key={code} value={code}>
              {UO_META[code].abbr} — {UO_META[code].name}
            </option>
          ))}
        </select>

        <span className="text-sm text-gray-500 ml-auto">
          {total} courses · Page {page}/{totalPages || 1}
        </span>
      </div>

      {/* Results table */}
      {loading ? (
        <div className="text-center py-12 text-gray-500">Loading...</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 text-left">
                <th className="py-2 px-3 font-medium text-gray-600 w-12">#</th>
                <th className="py-2 px-3 font-medium text-gray-600">Text</th>
                <th className="py-2 px-3 font-medium text-gray-600 w-32">Gold</th>
                <th className="py-2 px-3 font-medium text-gray-600 w-32">TF-IDF</th>
                <th className="py-2 px-3 font-medium text-gray-600 w-32">BERT</th>
                <th className="py-2 px-3 font-medium text-gray-600 w-16">Match</th>
              </tr>
            </thead>
            <tbody>
              {courses.map((course, i) => {
                const tfidfMatch = checkMatch(course.labels_uo, course.tfidf_pred);
                const bertMatch = checkMatch(course.labels_uo, course.bert_binary_pred);
                const isExpanded = expandedIdx === i;

                return (
                  <tr
                    key={i}
                    className="border-b border-gray-100 hover:bg-gray-50 cursor-pointer"
                    onClick={() => setExpandedIdx(isExpanded ? null : i)}
                  >
                    <td className="py-2.5 px-3 text-gray-400 font-mono text-xs">
                      {(page - 1) * perPage + i + 1}
                    </td>
                    <td className="py-2.5 px-3">
                      <p className={`text-gray-700 ${isExpanded ? "" : "line-clamp-2"}`}>
                        {isExpanded ? course.text_full || course.text : course.text}
                      </p>
                      {isExpanded && course.bert_dist_pct && (
                        <div className="mt-3 p-3 bg-gray-50 rounded-lg">
                          <UODistributionChart
                            predictions={course.bert_dist_pct}
                            type="distributional"
                            title="BERT Distributional Prediction"
                            height={220}
                          />
                        </div>
                      )}
                    </td>
                    <td className="py-2.5 px-3">
                      <div className="flex flex-wrap gap-1">
                        {getLabelChips(course.labels_uo)}
                      </div>
                    </td>
                    <td className="py-2.5 px-3">
                      <div className="flex flex-wrap gap-1">
                        {getPredChips(course.tfidf_pred)}
                      </div>
                    </td>
                    <td className="py-2.5 px-3">
                      <div className="flex flex-wrap gap-1">
                        {getPredChips(course.bert_binary_pred)}
                      </div>
                    </td>
                    <td className="py-2.5 px-3 text-center">
                      <div className="flex gap-1 justify-center">
                        {tfidfMatch === true && <span title="TF-IDF correct">✅</span>}
                        {tfidfMatch === false && <span title="TF-IDF wrong">❌</span>}
                        {bertMatch === true && <span title="BERT correct">✅</span>}
                        {bertMatch === false && <span title="BERT wrong">❌</span>}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 mt-6">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page === 1}
            className="px-3 py-1.5 border border-gray-300 rounded text-sm disabled:opacity-50"
          >
            Previous
          </button>
          <span className="text-sm text-gray-600">
            Page {page} of {totalPages}
          </span>
          <button
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
            className="px-3 py-1.5 border border-gray-300 rounded text-sm disabled:opacity-50"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
