\\documentclass\[aspectratio=169, 10pt\]{beamer}

\\usepackage{fontspec} \\usepackage\[vietnamese, english, bidi=basic, provide=\*\]{babel} \\babelprovide\[import, main, onchar=ids fonts\]{vietnamese} \\babelprovide\[import\]{english} \\babelfont{rm}{Noto Sans} \\babelfont{sf}{Noto Sans}

\\usepackage{amsmath} \\usepackage{amsfonts} \\usepackage{amssymb} \\usepackage{booktabs} \\usepackage{tabularx} \\usepackage{xcolor} \\usepackage{tikz} \\usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc}

% --- Professional Academic Theme Settings --- \\usetheme{Boadilla} \\usecolortheme{rose}

\\definecolor{navyprimary}{RGB}{15, 44, 89} \\definecolor{slateaccent}{RGB}{51, 65, 85} \\definecolor{blueaccent}{RGB}{29, 78, 216} \\definecolor{lightgraybg}{RGB}{248, 250, 252} \\definecolor{darktext}{RGB}{30, 41, 59} \\definecolor{mathred}{RGB}{185, 28, 28} \\definecolor{mathgreen}{RGB}{22, 101, 52}

\\setbeamercolor{structure}{fg=navyprimary} \\setbeamercolor{title}{fg=white, bg=navyprimary} \\setbeamercolor{frametitle}{fg=white, bg=navyprimary} \\setbeamercolor{block title}{fg=white, bg=navyprimary} \\setbeamercolor{block body}{fg=darktext, bg=lightgraybg} \\setbeamercolor{block title example}{fg=white, bg=slateaccent} \\setbeamercolor{block body example}{fg=darktext, bg=lightgraybg}

\\setbeamertemplate{navigation symbols}{} \\setbeamertemplate{footline}\[frame number\]

% --- Metadata --- \\title\[Kiến trúc Chuyên sâu ColPali\]{\\textbf{Kiến trúc Chuyên sâu ColPali:\\Efficient Document Retrieval with Vision Language Models}} \\subtitle{Bóc tách Toán học, Biểu diễn Tensor, Cơ chế Huấn luyện & Pipeline Triển khai} \\author\[Báo cáo Chuyên sâu AI\]{Báo cáo Kỹ thuật Chuyên đề: Vision-Language Models in IR} \\institute\[Advanced AI Lab\]{Phòng Thí nghiệm Trí tuệ Nhân tạo & Xử lý Ngôn ngữ Tự nhiên} \\date{\\today}

\\begin{document}

% ==================== SLIDE 1: TITLE ==================== \\begin{frame} \\titlepage \\end{frame}

% ==================== SLIDE 2: MỤC LỤC CHI TIẾT ==================== \\begin{frame}{Nội dung Báo cáo Chuyên sâu} \\begin{columns}\[T\] \\begin{column}{0.48\\textwidth} \\begin{block}{Phần I: Giới hạn của Pipeline Parsing} \\begin{itemize} \\item Phân tích sai số tích lũy (Error Cascading) \\item Mất mát thông tin cấu trúc 2 chiều \\end{itemize} \\end{block} \\begin{block}{Phần II: Cấu trúc Mô hình ColPali} \\begin{itemize} \\item SigLIP Vision Encoder & Patch Tokenization \\item Linear Projection & Gemma-2B Aut