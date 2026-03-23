import "./globals.css";
import type { Metadata } from "next";
import { ReactNode } from "react";

export const metadata: Metadata = {
  title: "Multi-Docs RAG (local Ollama + LangChain + LangGraph)",
  description: "Local multi-document RAG with Next.js, LangChain.js and LangGraph.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
