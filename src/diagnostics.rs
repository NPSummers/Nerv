//! Lightweight diagnostic utilities for human-friendly error reporting.
//!
//! Responsibilities:
//! - Convert byte offsets into line/column for caret diagnostics.
//! - Render compact, colorized error messages without external crates.
//! - Keep footprint minimal so diagnostics are easily reusable in all stages.
//!
pub struct Span {
    pub start: usize,
    pub end: usize,
}

pub struct SourceFile<'a> {
    pub name: &'a str,
    pub src: &'a str,
}

pub fn render_error(message: &str, source: &SourceFile, span: Option<&Span>) -> String {
    match span {
        None => format!("error: {}", message),
        Some(s) => {
            let (line, col) = byte_to_line_col(source.src, s.start);
            let line_text = line_text(source.src, line);
            let gutter = format!("{}:{}:", source.name, line);
            let header = format!("\x1b[31merror\x1b[0m: {}\n{}{} {}", message, gutter, col, "");

            let num_width = line.to_string().len();
            let line_num = format!("{:>width$} |", line, width = num_width);
            let caret_pad = " ".repeat(col.saturating_sub(1));
            let underline_len = s.end.saturating_sub(s.start).max(1);
            let underline = "^".to_string() + &"~".repeat(underline_len.saturating_sub(1));

            format!(
                "{}\n{} {}\n{} \x1b[31m{}\x1b[0m",
                header,
                line_num,
                line_text,
                " ".repeat(num_width) + " |",
                caret_pad + &underline,
            )
        }
    }
}

fn byte_to_line_col(src: &str, byte_pos: usize) -> (usize, usize) {
    let mut line = 1usize;
    let mut col = 1usize;
    let mut count = 0usize;
    for ch in src.chars() {
        let ch_len = ch.len_utf8();
        if count >= byte_pos { break; }
        if ch == '\n' { line += 1; col = 1; } else { col += 1; }
        count += ch_len;
    }
    (line, col)
}

fn line_text(src: &str, line: usize) -> String {
    src.lines().nth(line.saturating_sub(1)).unwrap_or("").to_string()
}


