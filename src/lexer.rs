use logos::Logos;

#[derive(Logos, Debug, PartialEq, Clone)]
pub enum Token {
    // Standard library function tokens (auto-generated via macro)
    #[token("printf")]
    Printf,
    #[token("puts")]
    Puts,
    #[token("malloc")]
    Malloc,
    #[token("free")]
    Free,
    #[token("abs")]
    Abs,
    #[token("sqrt")]
    Sqrt,
    #[token("pow")]
    Pow,

    // Keywords
    #[token("plug")]
    Plug,
    #[token("class")]
    Class,
    #[token("return")]
    Return,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("in")]
    In,
    #[token("for")]
    For,
    #[token("while")]
    While,
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[token("import")]
    Import,
    #[token("from")]
    From,
    #[token("mutable")]
    Mutable,
    #[token("this")]
    This,
    #[token("print")]
    Print,

    // Types
    #[token("int")]
    IntType,
    #[token("float")]
    FloatType,
    #[token("bool")]
    BoolType,
    #[token("string")]
    StringType,
    #[token("char")]
    CharType,
    #[token("list")]
    ListType,
    #[token("dict")]
    DictType,

    // Identifier
    #[regex("[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    Identifier(String),

    // Literals
    #[regex("[0-9]+", |lex| lex.slice().parse().ok())]
    Integer(i64),
    #[regex("[0-9]+\\.[0-9]+", |lex| lex.slice().parse().ok())]
    Float(f64),
    #[regex(r#""([^"\\]|\\.)*""#, |lex| Some(lex.slice()[1..lex.slice().len()-1].to_string()))]
    String(String),
    #[regex(r#"'([^'\\]|\\.)*'"#, |lex| lex.slice()[1..lex.slice().len()-1].chars().next())]
    Char(char),

    // Operators
    #[token("=")]
    Assign,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("==")]
    Equal,
    #[token("!=")]
    NotEqual,
    #[token("<")]
    LessThan,
    #[token(">")]
    GreaterThan,
    #[token("<=")]
    LessThanOrEqual,
    #[token(">=")]
    GreaterThanOrEqual,
    
    // Logical operators
    #[token("&&")]
    And,
    #[token("||")]
    Or,
    #[token("!")]
    Not,
    
    // Bitwise operators
    #[token("&")]
    BitwiseAnd,
    #[token("|")]
    BitwiseOr,
    #[token("^")]
    BitwiseXor,
    #[token("<<")]
    LeftShift,
    #[token(">>")]
    RightShift,

    // Delimiters
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token(",")]
    Comma,
    #[token(":")]
    Colon,
    #[token(".")]
    Dot,
    #[token(";")]
    Semicolon,

    // Comments
    #[regex("#[^\n]*", logos::skip)]
    SingleLineComment,
    #[regex("#\\{", multi_line_comment)]
    MultiLineComment,

    // Whitespace
    #[regex(r"[ \t\n\f]+", logos::skip)]
    Whitespace,

    #[error]
    Error,
}

fn multi_line_comment(lexer: &mut logos::Lexer<Token>) -> logos::Skip {
    let mut depth = 1;
    let mut remainder = lexer.remainder();
    let mut len = 0;

    while depth > 0 {
        if remainder.starts_with("#{") {
            depth += 1;
            remainder = &remainder[2..];
            len += 2;
        } else if remainder.starts_with("}#") {
            depth -= 1;
            remainder = &remainder[2..];
            len += 2;
        } else if !remainder.is_empty() {
            remainder = &remainder[1..];
            len += 1;
        } else {
            break; // End of file
        }
    }
    lexer.bump(len);
    logos::Skip
}

// Auto-generated stdlib token helper functions
crate::stdlib_token_helpers!();
