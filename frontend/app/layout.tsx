
import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "RoadVision AI | Road Damage Detection",
  description: "AI-powered automated road damage detection and reporting using YOLO11s."
};

export default function RootLayout({
  children
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
