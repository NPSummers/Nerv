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
            Some(Token::For) => self.parse_for_statement(),
            Some(Token::Plug) => {
                // Look ahead to see if this is a function or variable declaration
                self.parse_plug_statement()
            },
            Some(Token::Class) => self.parse_class_statement(),
            Some(Token::Return) => self.parse_return_statement(),
            Some(Token::If) => self.parse_if_statement(),
            Some(Token::While) => self.parse_while_statement(),
            Some(Token::Print) => self.parse_print_statement(),
            Some(Token::Import) => self.parse_import_statement(),
            Some(Token::From) => self.parse_from_import_statement(),
            _ => {
                let expr = self.parse_expression(0)?;
                if !matches!(self.peek(), Some(Token::Semicolon)) {
                    return Err(format!("Expected semicolon after expression, found {:?}", self.peek()));
                }
                self.consume(Token::Semicolon)?;
                Ok(Stmt::Expr(expr))
            }
        }
    }

    fn parse_for_statement(&mut self) -> Result<Stmt, String> {
        // Syntax: for (i in range(start, end[, step])) { ... }
        self.consume(Token::For)?;
        self.consume(Token::LParen)?;
        let var_name = self.consume_identifier()?;
        self.consume(Token::In)?;
        let iterable = self.parse_expression(0)?;
        self.consume(Token::RParen)?;

        self.consume(Token::LBrace)?;
        let mut body = Vec::new();
        while !matches!(self.peek(), Some(Token::RBrace)) {
            body.push(self.parse_statement()?);
        }
        self.consume(Token::RBrace)?;

        Ok(Stmt::For(ForStmt { variable: var_name, iterable, body }))
    }


    fn parse_expression(&mut self, precedence: u8) -> Result<Expr, String> {
        let mut left = self.parse_prefix()?;

        while precedence < self.get_peek_precedence() {
            // Check for member access first (highest precedence)
            if matches!(self.peek(), Some(Token::Dot)) {
                self.next(); // Consume '.'
                let member = self.consume_identifier()?;
                
                // Check if this is followed by parentheses (method call)
                if matches!(self.peek(), Some(Token::LParen)) {
                    // This is a method call
                    self.consume(Token::LParen)?;
                    let mut args = Vec::new();
                    
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
                    
                    // Create a special function call with object context
                    left = Expr::FunctionCall(FunctionCallExpr {
                        name: format!("{}::{}", "METHOD_CALL", member), // Special marker for method calls
                        args: {
                            let mut method_args = vec![left]; // First arg is the object
                            method_args.extend(args);
                            method_args
                        },
                    });
                } else {
                    // Regular member access
                    left = Expr::MemberAccess(MemberAccessExpr {
                        object: Box::new(left),
                        member,
                    });
                }
                continue;
            }
            
            // Check for assignment (special case)
            if matches!(self.peek(), Some(Token::Assign)) {
                match left {
                    Expr::Identifier(name) => {
                        self.next(); // Consume '='
                        let value = self.parse_expression(0)?; // Right-associative, low precedence
                        return Ok(Expr::Assignment(AssignmentExpr {
                            target: name,
                            value: Box::new(value),
                        }));
                    }
                    Expr::MemberAccess(member_access) => {
                        // Handle member assignment
                        self.next(); // Consume '='
                        let value = self.parse_expression(0)?;
                        
                        // Create a special assignment expression for member access
                        // We'll use a function call with a special name to represent this
                        return Ok(Expr::FunctionCall(FunctionCallExpr {
                            name: "MEMBER_ASSIGN".to_string(),
                            args: vec![
                                Expr::MemberAccess(member_access.clone()),
                                value
                            ],
                        }));
                    }
                    _ => {
                        return Err("Invalid assignment target".to_string());
                    }
                }
            }
            
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
            Token::This => Ok(Expr::Identifier("this".to_string())),
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
        // Check if this is followed by a parenthesis (function call or object instantiation)
        if let Some(Token::LParen) = self.peek() {
            // For now, we'll determine if it's a class instantiation during semantic analysis
            // In the parser, we'll treat class instantiation as function calls
            // and distinguish them in the codegen phase
            self.parse_function_call_or_instantiation(name)
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

    fn parse_function_call_or_instantiation(&mut self, name: String) -> Result<Expr, String> {
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
        
        // For now, return as FunctionCall and we'll distinguish in codegen
        // based on whether the name refers to a class or function
        Ok(Expr::FunctionCall(FunctionCallExpr { name, args }))
    }

    fn get_peek_precedence(&mut self) -> u8 {
        self.peek().map_or(0, Self::get_op_precedence_from_token)
    }

    fn get_op_precedence_from_token(token: &Token) -> u8 {
        match token {
            Token::Assign => 1,  // Assignment has very low precedence
            Token::Or => 2,
            Token::And => 3,
            Token::BitwiseOr => 4,
            Token::BitwiseXor => 5,
            Token::BitwiseAnd => 6,
            Token::Equal | Token::NotEqual => 7,
            Token::LessThan | Token::GreaterThan | Token::LessThanOrEqual | Token::GreaterThanOrEqual => 8,
            Token::LeftShift | Token::RightShift => 9,
            Token::Plus | Token::Minus => 10,
            Token::Star | Token::Slash | Token::Percent => 11,
            Token::Dot => 12,  // Member access has very high precedence
            _ => 0,
        }
    }
    
    fn get_op_precedence(&self, op: &BinaryOp) -> u8 {
        match op {
            BinaryOp::Or => 2,
            BinaryOp::And => 3,
            BinaryOp::BitwiseOr => 4,
            BinaryOp::BitwiseXor => 5,
            BinaryOp::BitwiseAnd => 6,
            BinaryOp::Equal | BinaryOp::NotEqual => 7,
            BinaryOp::LessThan | BinaryOp::GreaterThan | BinaryOp::LessThanOrEqual | BinaryOp::GreaterThanOrEqual => 8,
            BinaryOp::LeftShift | BinaryOp::RightShift => 9,
            BinaryOp::Add | BinaryOp::Subtract => 10,
            BinaryOp::Multiply | BinaryOp::Divide | BinaryOp::Modulo => 11,
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
        // LBracket is already consumed by the caller
        
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
        // LBrace is already consumed by the caller
        
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

    fn parse_import_statement(&mut self) -> Result<Stmt, String> {
        self.consume(Token::Import)?;
        let module = if let Some(Token::Identifier(name)) = self.next() {
            name
        } else {
            return Err("Expected module name after 'import'".to_string());
        };
        self.consume(Token::Semicolon)?;
        Ok(Stmt::Import(ImportStmt { module, items: None }))
    }

    fn parse_from_import_statement(&mut self) -> Result<Stmt, String> {
        self.consume(Token::From)?;
        let module = if let Some(Token::Identifier(name)) = self.next() {
            name
        } else {
            return Err("Expected module name after 'from'".to_string());
        };
        self.consume(Token::Import)?;
        
        let mut items = Vec::new();
        loop {
            if let Some(Token::Identifier(name)) = self.next() {
                items.push(name);
            } else {
                return Err("Expected identifier in import list".to_string());
            }
            
            if matches!(self.peek(), Some(Token::Comma)) {
                self.consume(Token::Comma)?;
            } else {
                break;
            }
        }
        
        self.consume(Token::Semicolon)?;
        Ok(Stmt::Import(ImportStmt { module, items: Some(items) }))
    }

    fn parse_class_statement(&mut self) -> Result<Stmt, String> {
        self.consume(Token::Class)?;
        let class_name = self.consume_identifier()?;
        
        self.consume(Token::LBrace)?;
        
        let mut members = Vec::new();
        let mut methods = Vec::new();
        
        while !matches!(self.peek(), Some(Token::RBrace)) {
            if matches!(self.peek(), Some(Token::Plug)) {
                // Parse member variable or method
                self.consume(Token::Plug)?;
                let name = self.consume_identifier()?;
                
                if matches!(self.peek(), Some(Token::LParen)) {
                    // This is a method (function)
                    let method = self.parse_function_declaration(name)?;
                    if let Stmt::FunctionDecl(func_decl) = method {
                        methods.push(func_decl);
                    }
                } else {
                    // This is a member variable
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
                    
                    members.push(VarDeclStmt {
                        name,
                        type_annotation,
                        initializer,
                        mutable: false, // Members are not mutable by default
                    });
                }
            } else {
                return Err("Expected 'plug' keyword for class members".to_string());
            }
        }
        
        self.consume(Token::RBrace)?;
        
        Ok(Stmt::ClassDecl(ClassDeclStmt {
            name: class_name,
            members,
            methods,
        }))
    }
}

fn tokens_match(a: &Token, b: &Token) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}
