import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import "./index.css";

// ------------------------------------------------------------------
// Catch unhandled promise rejections from Cesium (failed tile/asset
// fetches emit RequestErrorEvent). Without this, the browser may
// treat them as uncaught errors that crash the page — which,
// combined with URL-persisted ?mode=assistant, creates an infinite
// reload loop.
// ------------------------------------------------------------------
window.addEventListener("unhandledrejection", (e) => {
  const reason = e.reason;
  // Cesium RequestErrorEvent or generic network fetch errors
  if (
    reason &&
    (reason.constructor?.name === "RequestErrorEvent" ||
      (typeof reason === "object" && "statusCode" in reason && "request" in reason))
  ) {
    e.preventDefault(); // swallow — it's just a failed tile/asset fetch
  }
});

ReactDOM.createRoot(document.getElementById("root")!).render(
    <BrowserRouter>
      <App />
    </BrowserRouter>
);
