import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { ShieldCheck, History, Home } from 'lucide-react';

const Navbar = () => {
    const location = useLocation();

    const isActive = (path) => location.pathname === path ? "text-indigo-400" : "text-slate-400 hover:text-slate-100";

    return (
        <nav className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50">
            <div className="container mx-auto px-4 h-16 flex items-center justify-between">
                <Link to="/" className="flex items-center gap-2 group">
                    <ShieldCheck className="w-8 h-8 text-indigo-500 transition-transform group-hover:scale-110" />
                    <span className="font-bold text-xl tracking-tight">Veri<span className="text-indigo-500">Scope</span></span>
                </Link>

                <div className="flex items-center gap-6">
                    <Link to="/" className={`flex items-center gap-1.5 text-sm font-medium transition-colors ${isActive('/')}`}>
                        <Home className="w-4 h-4" />
                        Analyze
                    </Link>
                    <Link to="/history" className={`flex items-center gap-1.5 text-sm font-medium transition-colors ${isActive('/history')}`}>
                        <History className="w-4 h-4" />
                        History
                    </Link>
                </div>
            </div>
        </nav>
    );
};

export default Navbar;
