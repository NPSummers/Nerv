#[derive(Debug, PartialEq)]
pub struct Program {
    pub body: Vec<Stmt>,
}

#[derive(Debug, PartialEq, Clone)]
#[allow(dead_code)]
pub enum Stmt {
    Expr(Expr),
    VarDecl(VarDeclStmt),
    FunctionDecl(FunctionDeclStmt),
    ClassDecl(ClassDeclStmt),
    Return(Option<Expr>),
    Print(Expr),
    If(IfStmt),
    While(WhileStmt),
    For(ForStmt),
    Import(ImportStmt),
}

#[derive(Debug, PartialEq, Clone)]
pub struct VarDeclStmt {
    pub name: String,
    pub type_annotation: Option<Type>,
    pub initializer: Option<Expr>,
    pub mutable: bool,
}

#[derive(Debug, PartialEq, Clone)]
pub struct FunctionDeclStmt {
    pub name: String,
    pub params: Vec<(String, Type)>, // (name, type)
    pub return_type: Option<Type>,
    pub body: Vec<Stmt>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ClassDeclStmt {
    pub name: String,
    pub members: Vec<VarDeclStmt>,
    pub methods: Vec<FunctionDeclStmt>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expr {
    Literal(LiteralExpr),
    Binary(BinaryExpr),
    Unary(UnaryExpr),
    Identifier(String),
    FunctionCall(FunctionCallExpr),
    Assignment(AssignmentExpr),
    MemberAccess(MemberAccessExpr),
    ObjectInstantiation(ObjectInstantiationExpr),
}

#[derive(Debug, PartialEq, Clone)]
pub struct FunctionCallExpr {
    pub name: String,
    pub args: Vec<Expr>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum LiteralExpr {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    Char(char),
    Array(Vec<Expr>),
    Dict(Vec<(Expr, Expr)>), // Key-value pairs
}

#[derive(Debug, PartialEq, Clone)]
pub struct BinaryExpr {
    pub op: BinaryOp,
    pub left: Box<Expr>,
    pub right: Box<Expr>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct UnaryExpr {
    pub op: UnaryOp,
    pub operand: Box<Expr>,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum BinaryOp {
    // Arithmetic
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    
    // Comparison
    Equal,
    NotEqual,
    LessThan,
    GreaterThan,
    LessThanOrEqual,
    GreaterThanOrEqual,
    
    // Logical
    And,
    Or,
    
    // Bitwise
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    LeftShift,
    RightShift,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum UnaryOp {
    Minus,
    Plus,
    Not, // Logical not
}

// Type system
#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    // Primitive types
    Int,
    Float,
    Bool,
    String,
    Char,
    
    // Collection types (keeping these as they're used in parser)
    List(Box<Type>),
    Dict(Box<Type>, Box<Type>), // Key type, Value type
    
    // Custom types
    Custom(String),
}

// Type implementation methods can be added back when needed for type checking

#[derive(Debug, PartialEq, Clone)]
pub struct AssignmentExpr {
    pub target: String,
    pub value: Box<Expr>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct IfStmt {
    pub condition: Expr,
    pub then_branch: Vec<Stmt>,
    pub else_branch: Option<Vec<Stmt>>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct WhileStmt {
    pub condition: Expr,
    pub body: Vec<Stmt>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ForStmt {
    pub variable: String,
    pub iterable: Expr,
    pub body: Vec<Stmt>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ImportStmt {
    pub module: String,
    pub items: Option<Vec<String>>, // None for "import module", Some(vec) for "from module import items"
}

#[derive(Debug, PartialEq, Clone)]
pub struct MemberAccessExpr {
    pub object: Box<Expr>,
    pub member: String,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ObjectInstantiationExpr {
    pub class_name: String,
    pub args: Vec<Expr>,
}

