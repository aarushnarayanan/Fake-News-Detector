import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { ArrowRight, Loader2, AlertCircle } from 'lucide-react';
import { motion } from 'framer-motion';

import { API_BASE_URL } from '../config';

const HomePage = () => {
    const [title, setTitle] = useState('');
    const [text, setText] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const navigate = useNavigate();

    const handleAnalyze = async () => {
        if (!text.trim()) return;

        setLoading(true);
        setError(null);
        try {
            // Logic to call API
            const response = await axios.post(`${API_BASE_URL}/analyze`, {
                title: title.trim() || null,
                text: text.trim()
            });

            // --- STEP 1: SAVE TO LOCAL STORAGE ---
            // Create a history item that matches what our UI needs
            const newHistoryItem = {
                id: response.data.id,
                title: title.trim() || null,
                text_preview: text.trim().substring(0, 100),
                label: response.data.label,
                probability: response.data.probability,
                created_at: new Date().toISOString() // Save current time
            };

            // Get existing history or start fresh
            const existingHistory = JSON.parse(localStorage.getItem('veriscope_history') || '[]');

            // Add new item to the top of the list
            const updatedHistory = [newHistoryItem, ...existingHistory].slice(0, 50); // Keep last 50

            // Save back to browser
            localStorage.setItem('veriscope_history', JSON.stringify(updatedHistory));
            // -------------------------------------

            navigate('/results', { state: { data: response.data } });
        } catch (err) {
            console.error(err);
            setError(err.response?.data?.detail || "Failed to analyze text. Ensure backend is running.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-4xl mx-auto mt-12 mb-20">
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="text-center mb-12"
            >
                <h1 className="text-4xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-white via-indigo-100 to-indigo-400 bg-clip-text text-transparent">
                    Detect Misinformation<br />with AI Precision
                </h1>
                <p className="text-slate-400 text-lg max-w-2xl mx-auto">
                    Provide the news title and body below to analyze its credibility using our advanced RoBERTa-based deep learning model.
                </p>
            </motion.div>

            <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.2, duration: 0.5 }}
                className="glass-card p-6 md:p-8"
            >
                <div className="mb-6">
                    <label className="block text-slate-400 text-sm font-medium mb-2 pl-1">Article Title (Optional)</label>
                    <input
                        type="text"
                        value={title}
                        onChange={(e) => setTitle(e.target.value)}
                        placeholder="e.g. Breaking: New discovery in deep space..."
                        className="w-full bg-slate-950/50 border border-slate-700 rounded-lg px-4 py-3 text-slate-100 placeholder:text-slate-600 focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none transition-all"
                    />
                </div>

                <div className="mb-2">
                    <label className="block text-slate-400 text-sm font-medium mb-2 pl-1">Article Body</label>
                    <textarea
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        maxLength={3000}
                        placeholder="Paste article body here..."
                        className="w-full h-64 bg-slate-950/50 border border-slate-700 rounded-lg p-4 text-slate-100 placeholder:text-slate-600 focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none resize-none transition-all"
                    />
                </div>

                <div className="mt-6 flex items-center justify-between">
                    <p className={`text-sm ${text.length >= 3000 ? 'text-red-400 font-bold' : 'text-slate-500'}`}>
                        {text.length} / 3000 characters
                    </p>
                    <button
                        onClick={handleAnalyze}
                        disabled={loading || !text.trim() || text.length > 3000}
                        className={`
                            px-8 py-3 rounded-lg font-semibold flex items-center gap-2 transition-all
                            ${loading || !text.trim() || text.length > 3000
                                ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                                : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-600/20 hover:shadow-indigo-600/40'}
                        `}
                    >
                        {loading ? (
                            <>
                                <Loader2 className="w-5 h-5 animate-spin" />
                                Analyzing...
                            </>
                        ) : (
                            <>
                                Analyze Text
                                <ArrowRight className="w-5 h-5" />
                            </>
                        )}
                    </button>
                </div>

                {error && (
                    <div className="mt-4 p-4 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center gap-3 text-red-400">
                        <AlertCircle className="w-5 h-5 flex-shrink-0" />
                        <p>{error}</p>
                    </div>
                )}
            </motion.div>
        </div>
    );
};

export default HomePage;
