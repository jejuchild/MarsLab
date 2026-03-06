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
const MarsNewsPage = lazy(() => import("./pages/MarsNewsPage"));
const MarsResearchPage = lazy(() => import("./pages/MarsResearchPage"));

const PageLoading = () => (
  <div className="h-screen w-screen flex items-center justify-center bg-[#0a0f18] text-[#6b7c9c]">
    Loading…
  </div>
);

/** Wrap lazy-loaded pages with their own error boundary + suspense */
function LazyPage({ children, scope }: { children: React.ReactNode; scope: string }) {
  return (
    <ErrorBoundary scope={scope}>
      <Suspense fallback={<PageLoading />}>
        {children}
      </Suspense>
    </ErrorBoundary>
  );
}

export default function App() {
  const { isOnline, isUpdateAvailable, cacheStats, updateApp } = useServiceWorker();

  return (
    <ErrorBoundary scope="App">
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
      <Suspense fallback={<PageLoading />}>
        <Routes>
          <Route path="/" element={<ErrorBoundary scope="MainPage"><MainPage /></ErrorBoundary>} />
          <Route path="/download" element={<LazyPage scope="DataDownload"><DataDownloadPage /></LazyPage>} />
          <Route path="/upload" element={<LazyPage scope="DataUpload"><DataUploadPage /></LazyPage>} />
          <Route path="/suggestions" element={<LazyPage scope="Suggestions"><FeatureSuggestionsPage /></LazyPage>} />
          <Route path="/discussions" element={<LazyPage scope="Discussions"><DailyDiscussionsPage /></LazyPage>} />
          <Route path="/news" element={<LazyPage scope="News"><MarsNewsPage /></LazyPage>} />
          <Route path="/research" element={<LazyPage scope="Research"><MarsResearchPage /></LazyPage>} />
        </Routes>
      </Suspense>
    </ErrorBoundary>
  );
}
