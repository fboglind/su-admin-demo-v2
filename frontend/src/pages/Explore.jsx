import { useState, useEffect } from "react";
import CourseCard from "../components/CourseCard";
import UODistributionChart from "../components/UODistributionChart";
import { apiFetch, UO_META, UO_CODES } from "../utils/uo";

export default function Explore() {
  const [courses, setCourses] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [split, setSplit] = useState("");
  const [uoFilter, setUoFilter] = useState("");
  const [searchText, setSearchText] = useState("");
  const [selectedCourse, setSelectedCourse] = useState(null);
  const [stats, setStats] = useState(null);

  const perPage = 20;

  useEffect(() => {
    loadStats();
  }, []);

  useEffect(() => {
    loadCourses();
  }, [page, split, uoFilter]);

  const loadStats = async () => {
    try {
      const data = await apiFetch("/api/stats");
      setStats(data);
    } catch (err) {
      console.error(err);
    }
  };

  const loadCourses = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ page, per_page: perPage });
      if (split) params.set("split", split);
      if (uoFilter) params.set("uo", uoFilter);
      if (searchText) params.set("q", searchText);

      const data = await apiFetch(`/api/corpus?${params}`);
      setCourses(data.courses);
      setTotal(data.total);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = () => {
    setPage(1);
    loadCourses();
  };

  const totalPages = Math.ceil(total / perPage);

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Corpus Explorer</h1>
        <p className="text-gray-500 mt-1">
          Browse and filter the full course corpus ({stats?.total?.toLocaleString() || "..."} course versions).
        </p>
      </div>

      {/* Stats overview */}
      {stats && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
          <div className="bg-white border border-gray-200 rounded-lg p-3">
            <p className="text-2xl font-bold text-gray-900">{stats.total?.toLocaleString()}</p>
            <p className="text-xs text-gray-500">Total courses</p>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-3">
            <p className="text-2xl font-bold text-gray-900">{stats.train?.toLocaleString()}</p>
            <p className="text-xs text-gray-500">Training set</p>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-3">
            <p className="text-2xl font-bold text-gray-900">{stats.val?.toLocaleString()}</p>
            <p className="text-xs text-gray-500">Validation set</p>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-3">
            <p className="text-2xl font-bold text-gray-900">10</p>
            <p className="text-xs text-gray-500">UO categories</p>
          </div>
        </div>
      )}

      {/* UO category chips as quick filters */}
      <div className="flex flex-wrap gap-2 mb-6">
        <button
          onClick={() => { setUoFilter(""); setPage(1); }}
          className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${
            !uoFilter
              ? "bg-gray-900 text-white"
              : "bg-gray-100 text-gray-600 hover:bg-gray-200"
          }`}
        >
          All
        </button>
        {UO_CODES.map((code) => {
          const uo = UO_META[code];
          const isActive = uoFilter === String(code);
          return (
            <button
              key={code}
              onClick={() => { setUoFilter(isActive ? "" : String(code)); setPage(1); }}
              className="px-3 py-1.5 rounded-full text-xs font-medium transition-colors"
              style={{
                backgroundColor: isActive ? uo.color : `${uo.color}15`,
                color: isActive ? "white" : uo.color,
              }}
            >
              {uo.abbr} — {uo.name}
              {stats?.[`count_${code}`] && (
                <span className="ml-1 opacity-75">({stats[`count_${code}`]})</span>
              )}
            </button>
          );
        })}
      </div>

      {/* Text search + split filter */}
      <div className="flex gap-3 mb-6">
        <input
          type="text"
          value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          placeholder="Filter by text content..."
          className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm"
        />
        <select
          value={split}
          onChange={(e) => { setSplit(e.target.value); setPage(1); }}
          className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
        >
          <option value="">All splits</option>
          <option value="train">Train</option>
          <option value="val">Validation</option>
        </select>
        <button
          onClick={handleSearch}
          className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg text-sm hover:bg-gray-200"
        >
          Filter
        </button>
      </div>

      {/* Results */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <p className="text-sm text-gray-500 mb-3">
            {total.toLocaleString()} courses · Page {page}/{totalPages || 1}
          </p>

          {loading ? (
            <div className="text-center py-12 text-gray-500">Loading...</div>
          ) : (
            <div className="flex flex-col gap-2">
              {courses.map((course, i) => (
                <CourseCard
                  key={i}
                  course={course}
                  onClick={setSelectedCourse}
                />
              ))}
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
                {page} / {totalPages}
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

        {/* Detail panel */}
        <div>
          {selectedCourse ? (
            <div className="sticky top-4 border border-gray-200 rounded-lg p-4 bg-white">
              <h3 className="font-semibold text-gray-800 mb-2">Course Detail</h3>
              {selectedCourse.id && (
                <p className="text-xs text-gray-400 mb-2 font-mono">ID: {selectedCourse.id}</p>
              )}
              <p className="text-sm text-gray-600 leading-relaxed mb-4 max-h-64 overflow-y-auto">
                {selectedCourse.text_full || selectedCourse.text}
              </p>

              {selectedCourse.bert_dist_pct && (
                <UODistributionChart
                  predictions={selectedCourse.bert_dist_pct}
                  type="distributional"
                  title="BERT Distribution"
                  height={240}
                />
              )}
              {selectedCourse.tfidf_pred && (
                <div className="mt-4">
                  <UODistributionChart
                    predictions={selectedCourse.tfidf_pred}
                    type="binary"
                    title="TF-IDF Prediction"
                    height={240}
                  />
                </div>
              )}
            </div>
          ) : (
            <div className="border border-dashed border-gray-300 rounded-lg p-6 text-center text-sm text-gray-400">
              Click a course to see details
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
