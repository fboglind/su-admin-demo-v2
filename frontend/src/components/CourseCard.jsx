import { getUO, parseLabels } from "../utils/uo";

/**
 * Compact card displaying a course with its gold labels and optional similarity score.
 */
export default function CourseCard({ course, onClick, showSimilarity = false }) {
  const goldLabels = parseLabels(course.labels_uo);

  return (
    <div
      className="border border-gray-200 rounded-lg p-4 hover:border-blue-300 hover:shadow-sm transition-all cursor-pointer bg-white"
      onClick={() => onClick?.(course)}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          {/* Text preview */}
          <p className="text-sm text-gray-800 line-clamp-3 leading-relaxed">
            {course.text || "No text available"}
          </p>

          {/* Gold labels */}
          <div className="flex flex-wrap gap-1.5 mt-2">
            {goldLabels.map((code) => {
              const uo = getUO(code);
              return (
                <span
                  key={code}
                  className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium text-white"
                  style={{ backgroundColor: uo.color }}
                >
                  {uo.abbr}
                </span>
              );
            })}
            {goldLabels.length === 0 && (
              <span className="text-xs text-gray-400">No gold labels</span>
            )}
          </div>
        </div>

        {/* Metadata sidebar */}
        <div className="flex flex-col items-end gap-1 shrink-0">
          {showSimilarity && course.similarity != null && (
            <span className="text-xs font-mono bg-green-50 text-green-700 px-2 py-0.5 rounded">
              {(course.similarity * 100).toFixed(1)}%
            </span>
          )}
          {course.split && (
            <span className="text-xs text-gray-400">
              {course.split}
            </span>
          )}
          {course.id && (
            <span className="text-xs text-gray-400 font-mono">
              #{course.id}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
