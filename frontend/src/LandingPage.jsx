
import React from "react";
import { Link } from "react-router-dom";

const LandingPage = () => {
  return (
    <div className="min-h-screen bg-gray-900 flex flex-col items-center justify-center text-center text-white px-6">
      <h1 className="text-4xl font-bold mb-4 text-indigo-400">
        Welcome to AI Anime Assistant
      </h1>
      <p className="text-gray-400 mb-8 max-w-md">
        Discover, manage, and track your favorite anime — powered by AI.
      </p>

      <div className="flex gap-4">
        <Link
          to="/login"
          className="px-6 py-3 bg-indigo-600 hover:bg-indigo-700 rounded-lg font-semibold transition"
        >
          Login
        </Link>

        <Link
          to="/signup"
          className="px-6 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg font-semibold transition"
        >
          Sign Up
        </Link>
      </div>

      <footer className="absolute bottom-6 text-gray-500 text-sm">
        © {new Date().getFullYear()} AI Anime Assistant
      </footer>
    </div>
  );
};

export default LandingPage;
