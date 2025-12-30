import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { ArrowRight, Loader2, AlertCircle, Link as LinkIcon, Globe, FileText } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

import { API_BASE_URL } from '../config';

const HomePage = () => {
    const [tab, setTab] = useState('text'); // 'text' or 'url'
    const [url, setUrl] = useState('');
    const [title, setTitle] = useState('');
    const [text, setText] = useState('');
    const [loading, setLoading] = useState(false);
    const [scraping, setScraping] = useState(false);
    const [error, setError] = useState(null);
    const navigate = useNavigate();

    const handleUrlScrape = async () => {
        if (!url.trim()) return;
        setScraping(true);
        setError(null);
        try {
            const res = await axios.post(`${API_BASE_URL}/scrape`, { url: url.trim() });
            // Immediately run analysis with the scraped data
            await executeAnalysis(res.data.title, res.data.text);
        } catch (err) {
            setError(err.response?.data?.detail || "Failed to fetch/analyze article from URL.");
        } finally {
            setScraping(false);
        }
    };

    const executeAnalysis = async (finalTitle, finalText) => {
        if (!finalText.trim()) return;

        setLoading(true);
        setError(null);
        try {
            const response = await axios.post(`${API_BASE_URL}/analyze`, {
                title: finalTitle?.trim() || null,
                text: finalText.trim()
            });

            const newHistoryItem = {
                ...response.data,
                title: finalTitle?.trim() || null,
                text_preview: finalText.trim().substring(0, 100),
                created_at: new Date().toISOString()
            };

            const existingHistory = JSON.parse(localStorage.getItem('veriscope_history') || '[]');
            const updatedHistory = [newHistoryItem, ...existingHistory].slice(0, 50);
            localStorage.setItem('veriscope_history', JSON.stringify(updatedHistory));

            navigate('/results', { state: { data: response.data } });
        } catch (err) {
            console.error(err);
            setError(err.response?.data?.detail || "Failed to analyze text. Ensure backend is running.");
        } finally {
            setLoading(false);
        }
    };

    const handleAnalyze = () => {
        executeAnalysis(title, text);
    };

    return (
        <div className="max-w-4xl mx-auto mt-12 mb-20 px-4">
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="text-center mb-8"
            >
                <h1 className="text-4xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-white via-indigo-100 to-indigo-400 bg-clip-text text-transparent">
                    Detect Misinformation<br />with AI Precision
                </h1>

                {/* --- TAB SWITCHER --- */}
                <div className="flex items-center justify-center gap-8 mb-8 relative">
                    <button
                        onClick={() => setTab('text')}
                        className={`relative py-2 px-4 transition-all duration-300 flex items-center gap-2 ${tab === 'text' ? 'text-indigo-400' : 'text-slate-500 hover:text-slate-300'}`}
                    >
                        <FileText className="w-4 h-4" />
                        <span className="font-semibold uppercase tracking-widest text-xs">Text Body</span>
                        {tab === 'text' && (
                            <motion.div
                                layoutId="activeTab"
                                className="absolute bottom-0 left-0 right-0 h-1 bg-indigo-500 rounded-full shadow-[0_0_15px_rgba(99,102,241,0.8)]"
                            />
                        )}
                    </button>

                    <button
                        onClick={() => setTab('url')}
                        className={`relative py-2 px-4 transition-all duration-300 flex items-center gap-2 ${tab === 'url' ? 'text-indigo-400' : 'text-slate-500 hover:text-slate-300'}`}
                    >
                        <Globe className="w-4 h-4" />
                        <span className="font-semibold uppercase tracking-widest text-xs">Article URL</span>
                        {tab === 'url' && (
                            <motion.div
                                layoutId="activeTab"
                                className="absolute bottom-0 left-0 right-0 h-1 bg-indigo-500 rounded-full shadow-[0_0_15px_rgba(99,102,241,0.8)]"
                            />
                        )}
                    </button>
                </div>
            </motion.div>

            <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.2, duration: 0.5 }}
                className="glass-card p-6 md:p-8"
            >
                <AnimatePresence mode="wait">
                    {tab === 'text' ? (
                        <motion.div
                            key="text"
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: 10 }}
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
                        </motion.div>
                    ) : (
                        <motion.div
                            key="url"
                            initial={{ opacity: 0, x: 10 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -10 }}
                        >
                            <div className="mb-6">
                                <label className="block text-slate-400 text-sm font-medium mb-2 pl-1 text-center mt-4">Paste news article link below</label>
                                <div className="relative group">
                                    <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                                        <LinkIcon className="h-5 w-5 text-indigo-500 group-focus-within:text-indigo-400 transition-colors" />
                                    </div>
                                    <input
                                        type="url"
                                        value={url}
                                        onChange={(e) => setUrl(e.target.value)}
                                        placeholder="https://www.example-news.com/article"
                                        className="w-full bg-slate-950/70 border-2 border-slate-800 rounded-xl pl-12 pr-4 py-5 text-slate-100 placeholder:text-slate-600 focus:border-indigo-500 outline-none transition-all text-lg shadow-inner"
                                    />
                                </div>
                                <p className="text-slate-500 text-xs mt-4 text-center">
                                    Our AI will automatically scan the webpage, extract the primary content, and prepare it for analysis.
                                </p>
                            </div>

                            <div className="flex justify-center mt-10">
                                <button
                                    onClick={handleUrlScrape}
                                    disabled={scraping || !url.trim()}
                                    className={`
                                        px-12 py-4 rounded-xl font-bold flex items-center gap-3 transition-all text-lg
                                        ${scraping || !url.trim()
                                            ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                                            : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-[0_0_20px_rgba(79,70,229,0.3)] hover:shadow-[0_0_30px_rgba(79,70,229,0.5)]'}
                                    `}
                                >
                                    {scraping ? (
                                        <>
                                            <Loader2 className="w-6 h-6 animate-spin" />
                                            Fetching Data...
                                        </>
                                    ) : (
                                        <>
                                            Get Article Content
                                            <ArrowRight className="w-6 h-6" />
                                        </>
                                    )}
                                </button>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

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
