import React, { useState } from 'react';
import { useLocation, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { CheckCircle, XCircle, ArrowLeft, MessageSquare } from 'lucide-react';
import axios from 'axios';
import { API_BASE_URL } from '../config';

const ResultsPage = () => {
    const location = useLocation();
    const data = location.state?.data;
    const [feedbackOpen, setFeedbackOpen] = useState(false);
    const [feedbackSent, setFeedbackSent] = useState(false);

    if (!data) {
        return (
            <div className="text-center mt-20">
                <p className="text-slate-400">No results found.</p>
                <Link to="/" className="text-indigo-400 hover:text-indigo-300 mt-4 inline-block">Go back home</Link>
            </div>
        );
    }

    const isFake = data.label === "FAKE";
    const confidence = (data.probability * 100).toFixed(1);

    // Function to parse highlighted text
    const renderHighlightedText = (text) => {
        if (!text) return <span className="text-slate-400 italic">Analysis details not available for this legacy entry.</span>;

        // Replace [[...]] with wrapped spans
        const parts = text.split(/\[\[(.*?)\]\]/g);

        return parts.map((part, i) => {
            // Even indices are normal text, odd are highlighted
            if (i % 2 === 1) {
                // Determine classes statically so Tailwind can find them
                const highlightClass = isFake
                    ? "bg-red-500/30 text-red-200 px-1 rounded"
                    : "bg-green-500/30 text-green-200 px-1 rounded";

                return (
                    <span key={i} className={highlightClass}>
                        {part}
                    </span>
                );
            }
            return <span key={i} className="text-slate-300">{part}</span>;
        });
    };
    const handleFeedback = async (label) => {
        try {
            await axios.post(`${API_BASE_URL}/feedback`, {
                prediction_id: data.id,
                user_label: label
            });
            setFeedbackSent(true);
            setTimeout(() => setFeedbackOpen(false), 2000);
        } catch (e) {
            console.error("Feedback failed", e);
        }
    };

    return (
        <div className="max-w-5xl mx-auto mt-8 mb-20">
            <Link to="/" className="inline-flex items-center text-slate-400 hover:text-white mb-6 transition-colors">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Analysis
            </Link>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Score Card */}
                <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="lg:col-span-1"
                >
                    <div className="glass-card p-6 text-center sticky top-24">
                        <div className={`
                            w-32 h-32 mx-auto rounded-full flex items-center justify-center border-4 mb-4
                            ${isFake ? 'border-red-500 bg-red-500/10' : 'border-green-500 bg-green-500/10'}
                        `}>
                            {isFake ? <XCircle className="w-16 h-16 text-red-500" /> : <CheckCircle className="w-16 h-16 text-green-500" />}
                        </div>

                        <h2 className="text-2xl font-bold mb-1">{data.label}</h2>
                        <p className={`text-3xl font-black mb-4 ${isFake ? 'text-red-400' : 'text-green-400'}`}>
                            {confidence}%
                        </p>
                        <p className="text-slate-500 text-sm mb-6">
                            Confidence Score
                        </p>

                        <div className="border-t border-slate-800 pt-6">
                            <h3 className="text-sm font-semibold text-slate-400 mb-3">Is this result incorrect?</h3>
                            {!feedbackOpen && !feedbackSent && (
                                <button
                                    onClick={() => setFeedbackOpen(true)}
                                    className="text-sm text-indigo-400 hover:text-indigo-300 flex items-center justify-center gap-2 mx-auto"
                                >
                                    <MessageSquare className="w-4 h-4" />
                                    Give Feedback
                                </button>
                            )}

                            {feedbackOpen && !feedbackSent && (
                                <div className="flex gap-2 justify-center">
                                    <button onClick={() => handleFeedback("REAL")} className="px-3 py-1 bg-green-900/50 text-green-400 rounded hover:bg-green-900 text-xs border border-green-800">
                                        Mark Real
                                    </button>
                                    <button onClick={() => handleFeedback("FAKE")} className="px-3 py-1 bg-red-900/50 text-red-400 rounded hover:bg-red-900 text-xs border border-red-800">
                                        Mark Fake
                                    </button>
                                </div>
                            )}

                            {feedbackSent && (
                                <p className="text-green-400 text-sm">Thanks for your feedback!</p>
                            )}
                        </div>
                    </div>
                </motion.div>

                {/* Content Card */}
                <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.1 }}
                    className="lg:col-span-2"
                >
                    <div className="glass-card p-8">
                        <h3 className="text-lg font-semibold mb-4 text-slate-200 flex items-center gap-2">
                            Analysis Highlights
                            <span className="text-xs font-normal px-2 py-0.5 rounded-full bg-slate-800 text-slate-400">
                                AI Evidence
                            </span>
                        </h3>

                        <div className="prose prose-invert prose-p:leading-relaxed max-w-none">
                            <p className="whitespace-pre-wrap text-slate-300">
                                {renderHighlightedText(data.highlighted_text)}
                            </p>
                        </div>

                        {data.evidence && data.evidence.length > 0 && (
                            <div className="mt-8 pt-6 border-t border-slate-800">
                                <h4 className="text-sm font-semibold text-slate-400 mb-4">Top Suspicious Snippets</h4>
                                <div className="space-y-3">
                                    {data.evidence.map((item, idx) => (
                                        <div key={idx} className="bg-slate-950/50 p-3 rounded-lg border border-slate-800 flex gap-3">
                                            <span className="text-xs font-mono text-slate-500 mt-1">{(item.score * 100).toFixed(0)}%</span>
                                            <p className="text-sm text-slate-300">{item.text}</p>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                </motion.div>
            </div>
        </div>
    );
};

export default ResultsPage;
