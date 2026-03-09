import { useState } from "react";
import CourseCard from "../components/CourseCard";
import UODistributionChart from "../components/UODistributionChart";
import { apiFetch } from "../utils/uo";

const EXAMPLE_QUERIES = [
  "maskininlärning och artificiell intelligens",
  "juridik avtalsrätt",
  "kemi laborationer",
  "pedagogik undervisning didaktik",
  "statistik och kvantitativ metod",
  "socialt arbete och välfärd",
];

export default function Search() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedCourse, setSelectedCourse] = useState(null);

  const handleSearch = async (q = query) => {
    const searchQuery = q.trim();
    if (!searchQuery) return;

    setLoading(true);
    setError(null);
    setSelectedCourse(null);

    try {
      const data = await apiFetch("/api/search", {
        method: "POST",
        body: JSON.stringify({ query: searchQuery, n: 20 }),
      });
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") handleSearch();
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Semantic Search</h1>
        <p className="text-gray-500 mt-1">
          Search course descriptions by meaning using KB-BERT embeddings and ChromaDB.
        </p>
      </div>

      {/* Search input */}
      <div className="flex gap-3 mb-4">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Sök kurser... (Search courses in Swedish or English)"
          className="flex-1 px-4 py-2.5 border border-gray-300 rounded-lg text-sm
                     focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        />
        <button
          onClick={() => handleSearch()}
          disabled={loading || !query.trim()}
          className="px-5 py-2.5 bg-blue-600 text-white rounded-lg font-medium text-sm
                     hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </div>

      {/* Example queries */}
      <div className="flex flex-wrap gap-2 mb-8">
        {EXAMPLE_QUERIES.map((eq) => (
          <button
            key={eq}
            onClick={() => {
              setQuery(eq);
              handleSearch(eq);
            }}
            className="px-3 py-1 border border-gray-200 rounded-full text-xs text-gray-600
                       hover:border-blue-300 hover:text-blue-700 hover:bg-blue-50 transition-colors"
          >
            {eq}
          </button>
        ))}
      </div>

      {error && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700 mb-4">
          {error}
        </div>
      )}

      {/* Results */}
      {results && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Result list */}
          <div className="lg:col-span-2">
            <p className="text-sm text-gray-500 mb-3">
              {results.total} results for "{results.query}"
            </p>
            <div className="flex flex-col gap-3">
              {results.results.map((course) => (
                <CourseCard
                  key={course.rank}
                  course={course}
                  showSimilarity
                  onClick={setSelectedCourse}
                />
              ))}
            </div>
          </div>

          {/* Detail panel */}
          <div className="lg:col-span-1">
            {selectedCourse ? (
              <div className="sticky top-4 border border-gray-200 rounded-lg p-4 bg-white">
                <h3 className="font-semibold text-gray-800 mb-2">Course Detail</h3>
                <p className="text-sm text-gray-600 leading-relaxed mb-4">
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
                Click a result to see prediction details
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
