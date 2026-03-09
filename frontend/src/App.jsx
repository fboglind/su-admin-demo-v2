import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Classify from "./pages/Classify";
import Compare from "./pages/Compare";
import Search from "./pages/Search";
import Explore from "./pages/Explore";

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50">
        <Navbar />
        <Routes>
          <Route path="/" element={<Classify />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/search" element={<Search />} />
          <Route path="/explore" element={<Explore />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
