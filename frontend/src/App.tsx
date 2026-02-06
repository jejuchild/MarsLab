import { lazy, Suspense } from "react";
import { Routes, Route } from "react-router-dom";
import MainPage from "./pages/MainPage";

// Lazy-load secondary pages (not needed on initial load)
const DataDownloadPage = lazy(() => import("./pages/DataDownloadPage"));
const DataUploadPage = lazy(() => import("./pages/DataUploadPage"));
const FeatureSuggestionsPage = lazy(() => import("./pages/FeatureSuggestionsPage"));

export default function App() {
  return (
    <Suspense fallback={<div className="h-screen w-screen flex items-center justify-center bg-[#0a0f18] text-[#6b7c9c]">Loading…</div>}>
      <Routes>
        <Route path="/" element={<MainPage />} />
        <Route path="/download" element={<DataDownloadPage />} />
        <Route path="/upload" element={<DataUploadPage />} />
        <Route path="/suggestions" element={<FeatureSuggestionsPage />} />
      </Routes>
    </Suspense>
  );
}
