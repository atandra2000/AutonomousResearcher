import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Research Engineer Console",
  description: "Authenticated operations console for autonomous research runs"
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
