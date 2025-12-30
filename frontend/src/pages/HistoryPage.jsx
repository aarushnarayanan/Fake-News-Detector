import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Clock, ExternalLink } from 'lucide-react';
import { motion } from 'framer-motion';
import { API_BASE_URL } from '../config';

const HistoryPage = () => {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchHistory();
    }, []);

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
        <div className="max-w-4xl mx-auto mt-12 mb-20">
            <h1 className="text-3xl font-bold mb-8 flex items-center gap-3">
                <Clock className="w-8 h-8 text-indigo-500" />
                Recent Analyses
            </h1>

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
                            className="glass-card p-4 flex items-center justify-between hover:bg-slate-900/80 transition-colors"
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
                                        {item.created_at}
                                    </span>
                                </div>
                                {item.title && (
                                    <h3 className="text-slate-100 font-semibold text-sm mb-1 line-clamp-1">
                                        {item.title}
                                    </h3>
                                )}
                                <p className="text-slate-400 text-xs line-clamp-1 max-w-xl italic">
                                    {item.text_preview}...
                                </p>
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
