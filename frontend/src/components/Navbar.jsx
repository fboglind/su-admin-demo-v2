import { NavLink } from "react-router-dom";

const links = [
  { to: "/", label: "Classify" },
  { to: "/compare", label: "Compare" },
  { to: "/search", label: "Search" },
  { to: "/explore", label: "Explore" },
];

export default function Navbar() {
  return (
    <nav className="bg-white border-b border-gray-200 px-6 py-3">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-lg font-bold text-gray-900">su-adm-cc</span>
          <span className="text-sm text-gray-500">Course Classifier</span>
        </div>

        <div className="flex gap-1">
          {links.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-blue-50 text-blue-700"
                    : "text-gray-600 hover:text-gray-900 hover:bg-gray-50"
                }`
              }
            >
              {label}
            </NavLink>
          ))}
        </div>
      </div>
    </nav>
  );
}
