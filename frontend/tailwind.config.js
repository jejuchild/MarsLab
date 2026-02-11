/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        primary: "#135bec",
        "bg-dark": "#0a0e17",
        "surface-dark": "#161e2d",
        "surface-elevated": "#1e2a3e",
        "border-dark": "#2d3a54",
        "border-subtle": "#1e293b",
        "text-primary": "#f1f5f9",
        "text-secondary": "#94a3b8",
        "text-muted": "#64748b",
        "status-success": "#22c55e",
        "status-warning": "#f59e0b",
        "status-error": "#ef4444",
        "status-info": "#3b82f6",
      },
      fontFamily: {
        display: ["Space Grotesk", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
