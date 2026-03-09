import { useState } from "react";
import UODistributionChart from "../components/UODistributionChart";
import CourseCard from "../components/CourseCard";
import { apiFetch } from "../utils/uo";

const EXAMPLE_TEXTS = [
  {
    label: "Computer Science (NA)",
    text: "Kursen behandlar grundläggande begrepp inom programmering och datastrukturer. Studenterna lär sig att designa och implementera algoritmer i Python, samt att analysera deras tidskomplexitet. Kursen omfattar listor, träd, grafer, sortering och sökning.",
  },
  {
    label: "Law (JU)",
    text: "Kursen ger en introduktion till det svenska rättssystemet med fokus på civilrätt och offentlig rätt. Studenterna studerar rättskällor, lagtolkning och juridisk argumentation. Kursen behandlar avtalsrätt, skadeståndsrätt och grundläggande förvaltningsrätt.",
  },
  {
    label: "Social Sciences (SA)",
    text: "Kursen introducerar centrala teorier och metoder inom statsvetenskap. Fokus ligger på demokrati, politiska institutioner och jämförande politik. Studenterna genomför en mindre forskningsuppgift med kvalitativ eller kvantitativ metod.",
  },
];

export default function Classify() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleClassify = async () => {
    if (!text.trim() || text.trim().length < 10) {
      setError("Please enter at least 10 characters.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await apiFetch("/api/classify", {
        method: "POST",
        body: JSON.stringify({ text: text.trim() }),
      });
      console.log("Classify response:", data);

      // Check if any models actually returned predictions
      const hasModels = data.models && Object.keys(data.models).length > 0;
      const hasMatches = data.nearest_matches?.length > 0;
      if (!hasModels && !hasMatches) {
        setError(
          "The API responded but no models returned predictions. " +
          "Check the Flask terminal — are models loaded at startup? " +
          (data.search_error ? `Search error: ${data.search_error}` : "")
        );
      }
      setResult(data);
    } catch (err) {
      console.error("Classify error:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadExample = (example) => {
    setText(example.text);
    setResult(null);
    setError(null);
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Classify a Course</h1>
        <p className="text-gray-500 mt-1">
          Paste a course description to predict its disciplinary domain (utbildningsområde).
          TF-IDF runs live; BERT predictions come from the nearest corpus match.
        </p>
      </div>

      {/* Input area */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Klistra in en kursbeskrivning här... (Paste a course description here)"
            className="w-full h-48 p-4 border border-gray-300 rounded-lg text-sm leading-relaxed
                       focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-y"
          />
          <div className="flex items-center justify-between mt-3">
            <span className="text-xs text-gray-400">
              {text.length} characters
            </span>
            <button
              onClick={handleClassify}
              disabled={loading || text.trim().length < 10}
              className="px-5 py-2 bg-blue-600 text-white rounded-lg font-medium text-sm
                         hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed
                         transition-colors"
            >
              {loading ? "Classifying..." : "Classify"}
            </button>
          </div>
        </div>

        {/* Examples sidebar */}
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2">Try an example</h3>
          <div className="flex flex-col gap-2">
            {EXAMPLE_TEXTS.map((ex, i) => (
              <button
                key={i}
                onClick={() => loadExample(ex)}
                className="text-left p-3 border border-gray-200 rounded-lg text-sm
                           hover:border-blue-300 hover:bg-blue-50 transition-colors"
              >
                <span className="font-medium text-gray-700">{ex.label}</span>
                <p className="text-gray-500 text-xs mt-1 line-clamp-2">{ex.text}</p>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="mt-8">
          <h2 className="text-lg font-bold text-gray-900 mb-4">Results</h2>

          {/* Model predictions side by side */}
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
            {/* No models loaded warning */}
            {result.models && Object.keys(result.models).length === 0 && (
              <div className="col-span-full p-4 bg-amber-50 border border-amber-200 rounded-lg text-sm text-amber-800">
                <p className="font-medium">No model predictions returned.</p>
                <p className="mt-1">
                  Check the Flask terminal for startup errors. Visit{" "}
                  <a href="/api/debug" target="_blank" className="underline font-mono">
                    /api/debug
                  </a>{" "}
                  to see what's loaded.
                </p>
                {result.search_error && (
                  <p className="mt-1 font-mono text-xs">Search error: {result.search_error}</p>
                )}
              </div>
            )}
            {result.models?.tfidf && (
              <div className="border border-gray-200 rounded-lg p-4 bg-white">
                <div className="flex items-center gap-2 mb-1">
                  <h3 className="font-semibold text-gray-800">TF-IDF Baseline</h3>
                  <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full">
                    live
                  </span>
                </div>
                <p className="text-xs text-gray-500 mb-3">Linear SVC on character n-grams</p>
                <UODistributionChart
                  predictions={result.models.tfidf.predictions}
                  type="binary"
                  height={260}
                />
              </div>
            )}

            {/* BERT binary (from nearest match) */}
            {result.models?.bert_binary && (
              <div className="border border-gray-200 rounded-lg p-4 bg-white">
                <div className="flex items-center gap-2 mb-1">
                  <h3 className="font-semibold text-gray-800">BERT Binary</h3>
                  <span className="text-xs bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full">
                    nearest match
                  </span>
                </div>
                <p className="text-xs text-gray-500 mb-3">
                  {result.models.bert_binary.source}
                </p>
                <UODistributionChart
                  predictions={result.models.bert_binary.predictions}
                  type="binary"
                  height={260}
                />
              </div>
            )}

            {/* BERT distributional (from nearest match) */}
            {result.models?.bert_dist && (
              <div className="border border-gray-200 rounded-lg p-4 bg-white">
                <div className="flex items-center gap-2 mb-1">
                  <h3 className="font-semibold text-gray-800">BERT Distribution</h3>
                  <span className="text-xs bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full">
                    nearest match
                  </span>
                </div>
                <p className="text-xs text-gray-500 mb-3">
                  {result.models.bert_dist.source}
                </p>
                <UODistributionChart
                  predictions={result.models.bert_dist.predictions}
                  type="distributional"
                  height={260}
                />
              </div>
            )}
          </div>

          {/* Nearest corpus matches */}
          {result.nearest_matches?.length > 0 && (
            <div className="mt-8">
              <h3 className="text-md font-semibold text-gray-800 mb-3">
                Nearest Corpus Matches (Semantic Similarity)
              </h3>
              <div className="flex flex-col gap-3">
                {result.nearest_matches.map((match) => (
                  <CourseCard
                    key={match.rank}
                    course={match.course}
                    showSimilarity
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
