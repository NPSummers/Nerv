use crate::ast::*;
use crate::lexer::Token;
use logos::{Lexer, Logos};
use std::iter::Peekable;

pub struct Parser<'a> {
    lexer: Peekable<Lexer<'a, Token>>,
}

impl<'a> Parser<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            lexer: Token::lexer(input).peekable(),
        }
    }

    pub fn parse_program(&mut self) -> Result<Program, String> {
        let mut body = Vec::new();
        while self.peek().is_some() {
            body.push(self.parse_statement()?);
        }
        Ok(Program { body })
    }

    fn parse_statement(&mut self) -> Result<Stmt, String> {
        match self.peek() {
            Some(Token::Plug) => {
                // Look ahead to see if this is a function or variable declaration
                self.parse_plug_statement()
            },
            Some(Token::Return) => self.parse_return_statement(),
            Some(Token::If) => self.parse_if_statement(),
            Some(Token::While) => self.parse_while_statement(),
            Some(Token::Print) => self.parse_print_statement(),
            _ => {
                let expr = self.parse_expression(0)?;
                self.consume(Token::Semicolon)?;
                Ok(Stmt::Expr(expr))
            }
        }
    }


    fn parse_expression(&mut self, precedence: u8) -> Result<Expr, String> {
        let mut left = self.parse_prefix()?;

        while precedence < self.get_peek_precedence() {
            let op = self.get_binary_op()?;
            let op_precedence = self.get_op_precedence(&op);
            self.next(); // Consume operator

            let right = self.parse_expression(op_precedence)?;
            left = Expr::Binary(BinaryExpr {
                op,
                left: Box::new(left),
                right: Box::new(right),
            });
        }

        Ok(left)
    }

    fn parse_prefix(&mut self) -> Result<Expr, String> {
        let token = self.next().ok_or_else(|| "Unexpected end of input".to_string())?;
        match token {
            Token::Integer(i) => Ok(Expr::Literal(LiteralExpr::Int(i))),
            Token::Float(f) => Ok(Expr::Literal(LiteralExpr::Float(f))),
            Token::String(s) => Ok(Expr::Literal(LiteralExpr::String(s))),
            Token::Char(c) => Ok(Expr::Literal(LiteralExpr::Char(c))),
            Token::True => Ok(Expr::Literal(LiteralExpr::Bool(true))),
            Token::False => Ok(Expr::Literal(LiteralExpr::Bool(false))),
            Token::Identifier(name) => self.parse_identifier_or_function_call(name),
            Token::Minus => {
                let operand = self.parse_expression(4)?; // High precedence for unary minus
                Ok(Expr::Unary(UnaryExpr {
                    op: UnaryOp::Minus,
                    operand: Box::new(operand),
                }))
            },
            Token::Plus => {
                let operand = self.parse_expression(4)?; // High precedence for unary plus
                Ok(Expr::Unary(UnaryExpr {
                    op: UnaryOp::Plus,
                    operand: Box::new(operand),
                }))
            },
            Token::Not => {
                let operand = self.parse_expression(4)?; // High precedence for unary not
                Ok(Expr::Unary(UnaryExpr {
                    op: UnaryOp::Not,
                    operand: Box::new(operand),
                }))
            },
            Token::LBracket => self.parse_array_literal(),
            Token::LBrace => self.parse_dict_literal(),
            Token::LParen => {
                let expr = self.parse_expression(0)?;
                self.consume(Token::RParen)?;
                Ok(expr)
            },
            token if token.is_stdlib_function() => {
                let name = token.as_function_name().unwrap().to_string();
                self.parse_function_call(name)
            },
            _ => Err(format!("Unexpected token {:?} for prefix expression", token)),
        }
    }

    fn parse_identifier_or_function_call(&mut self, name: String) -> Result<Expr, String> {
        // Check if this is followed by a parenthesis (function call)
        if let Some(Token::LParen) = self.peek() {
            self.parse_function_call(name)
        } else {
            Ok(Expr::Identifier(name))
        }
    }

    fn parse_function_call(&mut self, name: String) -> Result<Expr, String> {
        self.consume(Token::LParen)?;
        
        let mut args = Vec::new();
        
        // Parse arguments if there are any
        if !matches!(self.peek(), Some(Token::RParen)) {
            loop {
                args.push(self.parse_expression(0)?);
                
                if self.maybe_consume(Token::Comma) {
                    continue;
                } else {
                    break;
                }
            }
        }
        
        self.consume(Token::RParen)?;
        
        Ok(Expr::FunctionCall(FunctionCallExpr { name, args }))
    }

    fn get_peek_precedence(&mut self) -> u8 {
        self.peek().map_or(0, Self::get_op_precedence_from_token)
    }

    fn get_op_precedence_from_token(token: &Token) -> u8 {
        match token {
            Token::Or => 1,
            Token::And => 2,
            Token::BitwiseOr => 3,
            Token::BitwiseXor => 4,
            Token::BitwiseAnd => 5,
            Token::Equal | Token::NotEqual => 6,
            Token::LessThan | Token::GreaterThan | Token::LessThanOrEqual | Token::GreaterThanOrEqual => 7,
            Token::LeftShift | Token::RightShift => 8,
            Token::Plus | Token::Minus => 9,
            Token::Star | Token::Slash | Token::Percent => 10,
            _ => 0,
        }
    }
    
    fn get_op_precedence(&self, op: &BinaryOp) -> u8 {
        match op {
            BinaryOp::Or => 1,
            BinaryOp::And => 2,
            BinaryOp::BitwiseOr => 3,
            BinaryOp::BitwiseXor => 4,
            BinaryOp::BitwiseAnd => 5,
            BinaryOp::Equal | BinaryOp::NotEqual => 6,
            BinaryOp::LessThan | BinaryOp::GreaterThan | BinaryOp::LessThanOrEqual | BinaryOp::GreaterThanOrEqual => 7,
            BinaryOp::LeftShift | BinaryOp::RightShift => 8,
            BinaryOp::Add | BinaryOp::Subtract => 9,
            BinaryOp::Multiply | BinaryOp::Divide | BinaryOp::Modulo => 10,
        }
    }

    fn get_binary_op(&mut self) -> Result<BinaryOp, String> {
        match self.peek() {
            Some(Token::Plus) => Ok(BinaryOp::Add),
            Some(Token::Minus) => Ok(BinaryOp::Subtract),
            Some(Token::Star) => Ok(BinaryOp::Multiply),
            Some(Token::Slash) => Ok(BinaryOp::Divide),
            Some(Token::Percent) => Ok(BinaryOp::Modulo),
            Some(Token::Equal) => Ok(BinaryOp::Equal),
            Some(Token::NotEqual) => Ok(BinaryOp::NotEqual),
            Some(Token::LessThan) => Ok(BinaryOp::LessThan),
            Some(Token::GreaterThan) => Ok(BinaryOp::GreaterThan),
            Some(Token::LessThanOrEqual) => Ok(BinaryOp::LessThanOrEqual),
            Some(Token::GreaterThanOrEqual) => Ok(BinaryOp::GreaterThanOrEqual),
            Some(Token::And) => Ok(BinaryOp::And),
            Some(Token::Or) => Ok(BinaryOp::Or),
            Some(Token::BitwiseAnd) => Ok(BinaryOp::BitwiseAnd),
            Some(Token::BitwiseOr) => Ok(BinaryOp::BitwiseOr),
            Some(Token::BitwiseXor) => Ok(BinaryOp::BitwiseXor),
            Some(Token::LeftShift) => Ok(BinaryOp::LeftShift),
            Some(Token::RightShift) => Ok(BinaryOp::RightShift),
            _ => Err("Invalid binary operator".to_string()),
        }
    }

    fn peek(&mut self) -> Option<&Token> {
        self.lexer.peek()
    }

    fn next(&mut self) -> Option<Token> {
        self.lexer.next()
    }

    fn consume(&mut self, expected: Token) -> Result<(), String> {
        match self.next() {
            Some(token) if tokens_match(&token, &expected) => Ok(()),
            Some(token) => Err(format!("Expected {:?}, found {:?}", expected, token)),
            None => Err(format!("Expected {:?}, found EOF", expected)),
        }
    }

    fn maybe_consume(&mut self, token_type: Token) -> bool {
        match self.peek() {
            Some(token) if tokens_match(token, &token_type) => {
                self.next();
                true
            }
            _ => false,
        }
    }

    fn consume_identifier(&mut self) -> Result<String, String> {
        match self.next() {
            Some(Token::Identifier(name)) => Ok(name),
            _ => Err("Expected an identifier".to_string()),
        }
    }

    fn consume_type(&mut self) -> Result<Type, String> {
        match self.next() {
            Some(Token::IntType) => Ok(Type::Int),
            Some(Token::FloatType) => Ok(Type::Float),
            Some(Token::BoolType) => Ok(Type::Bool),
            Some(Token::StringType) => Ok(Type::String),
            Some(Token::CharType) => Ok(Type::Char),
            Some(Token::ListType) => {
                self.consume(Token::LessThan)?;
                let element_type = self.consume_type()?;
                self.consume(Token::GreaterThan)?;
                Ok(Type::List(Box::new(element_type)))
            },
            Some(Token::DictType) => {
                self.consume(Token::LessThan)?;
                let key_type = self.consume_type()?;
                self.consume(Token::Comma)?;
                let value_type = self.consume_type()?;
                self.consume(Token::GreaterThan)?;
                Ok(Type::Dict(Box::new(key_type), Box::new(value_type)))
            },
            Some(Token::Identifier(name)) => Ok(Type::Custom(name)),
            _ => Err("Expected a type".to_string()),
        }
    }

    fn parse_array_literal(&mut self) -> Result<Expr, String> {
        self.consume(Token::LBracket)?;
        
        let mut elements = Vec::new();
        
        // Parse elements if there are any
        if !matches!(self.peek(), Some(Token::RBracket)) {
            loop {
                elements.push(self.parse_expression(0)?);
                
                if self.maybe_consume(Token::Comma) {
                    continue;
                } else {
                    break;
                }
            }
        }
        
        self.consume(Token::RBracket)?;
        Ok(Expr::Literal(LiteralExpr::Array(elements)))
    }

    fn parse_dict_literal(&mut self) -> Result<Expr, String> {
        self.consume(Token::LBrace)?;
        
        let mut pairs = Vec::new();
        
        // Parse key-value pairs if there are any
        if !matches!(self.peek(), Some(Token::RBrace)) {
            loop {
                let key = self.parse_expression(0)?;
                self.consume(Token::Colon)?;
                let value = self.parse_expression(0)?;
                pairs.push((key, value));
                
                if self.maybe_consume(Token::Comma) {
                    continue;
                } else {
                    break;
                }
            }
        }
        
        self.consume(Token::RBrace)?;
        Ok(Expr::Literal(LiteralExpr::Dict(pairs)))
    }

    fn parse_plug_statement(&mut self) -> Result<Stmt, String> {
        self.consume(Token::Plug)?; // Consume 'plug'

        let mutable = self.maybe_consume(Token::Mutable);
        let name = self.consume_identifier()?;

        // Check if this is a function declaration (has parentheses)
        if let Some(Token::LParen) = self.peek() {
            // Parse function declaration
            self.parse_function_declaration(name)
        } else {
            // Parse variable declaration
            let type_annotation = if self.maybe_consume(Token::Colon) {
                Some(self.consume_type()?)
            } else {
                None
            };

            let initializer = if self.maybe_consume(Token::Assign) {
                Some(self.parse_expression(0)?)
            } else {
                None
            };

            self.consume(Token::Semicolon)?;

            Ok(Stmt::VarDecl(VarDeclStmt {
                name,
                type_annotation,
                initializer,
                mutable,
            }))
        }
    }

    fn parse_function_declaration(&mut self, name: String) -> Result<Stmt, String> {
        self.consume(Token::LParen)?;
        
        let mut params = Vec::new();
        
        // Parse parameters if there are any
        if !matches!(self.peek(), Some(Token::RParen)) {
            loop {
                let param_name = self.consume_identifier()?;
                self.consume(Token::Colon)?;
                let param_type = self.consume_type()?;
                params.push((param_name, param_type));
                
                if self.maybe_consume(Token::Comma) {
                    continue;
                } else {
                    break;
                }
            }
        }
        
        self.consume(Token::RParen)?;
        
        // Parse return type
        let return_type = if self.maybe_consume(Token::Colon) {
            Some(self.consume_type()?)
        } else {
            None
        };
        
        // Parse function body
        self.consume(Token::LBrace)?;
        let mut body = Vec::new();
        
        while !matches!(self.peek(), Some(Token::RBrace)) {
            body.push(self.parse_statement()?);
        }
        
        self.consume(Token::RBrace)?;
        
        Ok(Stmt::FunctionDecl(FunctionDeclStmt {
            name,
            params,
            return_type,
            body,
        }))
    }

    fn parse_return_statement(&mut self) -> Result<Stmt, String> {
        self.consume(Token::Return)?;
        
        let expr = if matches!(self.peek(), Some(Token::Semicolon)) {
            None
        } else {
            Some(self.parse_expression(0)?)
        };
        
        self.consume(Token::Semicolon)?;
        Ok(Stmt::Return(expr))
    }

    fn parse_if_statement(&mut self) -> Result<Stmt, String> {
        self.consume(Token::If)?;
        self.consume(Token::LParen)?;
        let condition = self.parse_expression(0)?;
        self.consume(Token::RParen)?;
        
        self.consume(Token::LBrace)?;
        let mut then_branch = Vec::new();
        while !matches!(self.peek(), Some(Token::RBrace)) {
            then_branch.push(self.parse_statement()?);
        }
        self.consume(Token::RBrace)?;
        
        let else_branch = if self.maybe_consume(Token::Else) {
            self.consume(Token::LBrace)?;
            let mut else_stmts = Vec::new();
            while !matches!(self.peek(), Some(Token::RBrace)) {
                else_stmts.push(self.parse_statement()?);
            }
            self.consume(Token::RBrace)?;
            Some(else_stmts)
        } else {
            None
        };
        
        Ok(Stmt::If(IfStmt {
            condition,
            then_branch,
            else_branch,
        }))
    }

    fn parse_while_statement(&mut self) -> Result<Stmt, String> {
        self.consume(Token::While)?;
        self.consume(Token::LParen)?;
        let condition = self.parse_expression(0)?;
        self.consume(Token::RParen)?;
        
        self.consume(Token::LBrace)?;
        let mut body = Vec::new();
        while !matches!(self.peek(), Some(Token::RBrace)) {
            body.push(self.parse_statement()?);
        }
        self.consume(Token::RBrace)?;
        
        Ok(Stmt::While(WhileStmt { condition, body }))
    }

    fn parse_print_statement(&mut self) -> Result<Stmt, String> {
        self.consume(Token::Print)?;
        self.consume(Token::LParen)?;
        let expr = self.parse_expression(0)?;
        self.consume(Token::RParen)?;
        self.consume(Token::Semicolon)?;
        Ok(Stmt::Print(expr))
    }
}

fn tokens_match(a: &Token, b: &Token) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}
