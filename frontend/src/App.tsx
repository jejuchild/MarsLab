import { Routes, Route } from "react-router-dom";
import MainPage from "./pages/MainPage";
import DataDownloadPage from "./pages/DataDownloadPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<MainPage />} />
      <Route path="/download" element={<DataDownloadPage />} />
    </Routes>
  );
}
