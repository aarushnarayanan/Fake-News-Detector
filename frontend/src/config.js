const isLocal = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";

// Toggle between your local Docker backend and your AWS Production backend automatically
export const API_BASE_URL = isLocal
    ? "http://localhost:8000"
    : "http://18.218.252.208:8000";
