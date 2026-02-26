import { lazy, Suspense } from "react";
import { Routes, Route } from "react-router-dom";
import { Toaster } from "react-hot-toast";
import MainPage from "./pages/MainPage";
import useServiceWorker from "./hooks/useServiceWorker";
import OfflineIndicator from "./components/OfflineIndicator";
import ErrorBoundary from "./components/ErrorBoundary";

// Lazy-load secondary pages (not needed on initial load)
const DataDownloadPage = lazy(() => import("./pages/DataDownloadPage"));
const DataUploadPage = lazy(() => import("./pages/DataUploadPage"));
const FeatureSuggestionsPage = lazy(() => import("./pages/FeatureSuggestionsPage"));
const DailyDiscussionsPage = lazy(() => import("./pages/DailyDiscussionsPage"));

export default function App() {
  const { isOnline, isUpdateAvailable, cacheStats, updateApp } = useServiceWorker();

  return (
    <ErrorBoundary>
      <OfflineIndicator
        isOnline={isOnline}
        isUpdateAvailable={isUpdateAvailable}
        cacheStats={cacheStats}
        onUpdate={updateApp}
      />
      <Toaster
        position="bottom-right"
        toastOptions={{
          style: {
            background: "#161e2d",
            color: "#f1f5f9",
            border: "1px solid #2d3a54",
            fontSize: "13px",
          },
          success: {
            iconTheme: { primary: "#22c55e", secondary: "#161e2d" },
          },
          error: {
            iconTheme: { primary: "#ef4444", secondary: "#161e2d" },
          },
        }}
      />
      <Suspense fallback={<div className="h-screen w-screen flex items-center justify-center bg-[#0a0f18] text-[#6b7c9c]">Loading…</div>}>
        <Routes>
          <Route path="/" element={<MainPage />} />
          <Route path="/download" element={<DataDownloadPage />} />
          <Route path="/upload" element={<DataUploadPage />} />
          <Route path="/suggestions" element={<FeatureSuggestionsPage />} />
          <Route path="/discussions" element={<DailyDiscussionsPage />} />
        </Routes>
      </Suspense>
    </ErrorBoundary>
  );
}
