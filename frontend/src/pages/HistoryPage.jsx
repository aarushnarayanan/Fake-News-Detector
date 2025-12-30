import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Clock, ExternalLink, Trash2, AlertCircle, CheckCircle, XCircle } from 'lucide-react';
import { motion } from 'framer-motion';
import { API_BASE_URL } from '../config';

const HistoryPage = () => {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);
    const [feedbackStatus, setFeedbackStatus] = useState({}); // { id: 'sent' }
    const navigate = useNavigate();

    useEffect(() => {
        fetchHistory();
    }, []);

    const handleFeedback = async (id, label, e) => {
        e.stopPropagation(); // Don't trigger the click-to-view report
        try {
            await axios.post(`${API_BASE_URL}/feedback`, {
                prediction_id: id,
                user_label: label
            });
            setFeedbackStatus(prev => ({ ...prev, [id]: 'sent' }));
        } catch (err) {
            console.error("Feedback failed", err);
        }
    };

    const clearHistory = () => {
        if (window.confirm("Are you sure you want to clear your local history? This cannot be undone.")) {
            localStorage.removeItem('veriscope_history');
            setHistory([]);
        }
    };

    const fetchHistory = () => {
        try {
            // --- STEP 2: READ FROM LOCAL STORAGE ---
            // Instead of call the API, we grab the data from the browser's memory
            const localData = localStorage.getItem('veriscope_history');

            if (localData) {
                setHistory(JSON.parse(localData));
            } else {
                setHistory([]);
            }
            // ----------------------------------------
        } catch (err) {
            console.error("Failed to load local history:", err);
            setHistory([]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-4xl mx-auto mt-12 mb-20 px-4">
            <div className="flex items-center justify-between mb-8">
                <h1 className="text-3xl font-bold flex items-center gap-3">
                    <Clock className="w-8 h-8 text-indigo-500" />
                    Recent Analyses
                </h1>
                {history.length > 0 && (
                    <button
                        onClick={clearHistory}
                        className="flex items-center gap-2 text-slate-500 hover:text-red-400 text-sm transition-colors border border-slate-800 hover:border-red-900/50 px-3 py-1.5 rounded-lg bg-slate-900/50"
                    >
                        <Trash2 className="w-4 h-4" />
                        Clear All
                    </button>
                )}
            </div>

            {loading ? (
                <div className="text-center text-slate-500">Loading history...</div>
            ) : (
                <div className="space-y-4">
                    {history.map((item, i) => (
                        <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: i * 0.05 }}
                            key={item.id}
                            onClick={() => navigate('/results', { state: { data: item } })}
                            className="glass-card p-4 flex items-center justify-between hover:bg-slate-900/80 hover:border-indigo-500/50 cursor-pointer transition-all group"
                        >
                            <div>
                                <div className="flex items-center gap-3 mb-2">
                                    <span className={`
                                        text-xs font-bold px-2 py-0.5 rounded uppercase
                                        ${item.label === 'FAKE' ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'}
                                    `}>
                                        {item.label}
                                    </span>
                                    <span className="text-xs text-slate-500">
                                        {(item.probability * 100).toFixed(1)}% Confidence
                                    </span>
                                    <span className="text-xs text-slate-600">
                                        {new Date(item.created_at).toLocaleDateString()}
                                    </span>
                                    {feedbackStatus[item.id] ? (
                                        <span className="text-[10px] text-green-500 font-bold uppercase tracking-wider">Feedback Sent</span>
                                    ) : (
                                        <div className="flex items-center gap-1 border-l border-slate-800 ml-1 pl-3">
                                            <button
                                                onClick={(e) => handleFeedback(item.id, 'REAL', e)}
                                                className="p-1.5 hover:bg-green-500/10 text-slate-500 hover:text-green-400 rounded-md transition-colors"
                                                title="Mark as Real"
                                            >
                                                <CheckCircle className="w-3.5 h-3.5" />
                                            </button>
                                            <button
                                                onClick={(e) => handleFeedback(item.id, 'FAKE', e)}
                                                className="p-1.5 hover:bg-red-500/10 text-slate-500 hover:text-red-400 rounded-md transition-colors"
                                                title="Mark as Fake"
                                            >
                                                <XCircle className="w-3.5 h-3.5" />
                                            </button>
                                        </div>
                                    )}
                                    {!item.highlighted_text && (
                                        <span className="flex items-center gap-1 text-[10px] font-bold px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-500 uppercase border border-amber-500/20">
                                            <AlertCircle className="w-3 h-3" />
                                            Legacy Item
                                        </span>
                                    )}
                                </div>
                                {item.title && (
                                    <h3 className="text-slate-100 font-semibold text-sm mb-1 line-clamp-1 group-hover:text-indigo-400 transition-colors">
                                        {item.title}
                                    </h3>
                                )}
                                <p className="text-slate-400 text-xs line-clamp-1 max-w-xl italic">
                                    {item.text_preview}...
                                </p>
                            </div>
                            <div className="flex items-center text-indigo-500 opacity-0 group-hover:opacity-100 transition-opacity">
                                <span className="text-xs font-medium mr-2">View Report</span>
                                <ExternalLink className="w-4 h-4" />
                            </div>
                        </motion.div>
                    ))}

                    {history.length === 0 && (
                        <div className="text-center py-12 text-slate-500 bg-slate-900/30 rounded-xl border border-dashed border-slate-800">
                            No history found. Start by analyzing some text!
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default HistoryPage;
