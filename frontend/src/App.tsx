import { Routes, Route } from "react-router-dom";
import MainPage from "./pages/MainPage";
import DataDownloadPage from "./pages/DataDownloadPage";
import DataUploadPage from "./pages/DataUploadPage";
import FeatureSuggestionsPage from "./pages/FeatureSuggestionsPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<MainPage />} />
      <Route path="/download" element={<DataDownloadPage />} />
      <Route path="/upload" element={<DataUploadPage />} />
      <Route path="/suggestions" element={<FeatureSuggestionsPage />} />
    </Routes>
  );
}
