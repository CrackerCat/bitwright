//! An and-inverter graph: every gate a conjunction of two literals, negation free on the
//! edges, with constant folding, trivial simplifications (`a ∧ a`, `a ∧ ¬a`) and structural
//! hashing, so equal subcircuits are one gate. Clauses are generated (Tseitin) only for the
//! gates a goal reaches.

use std::collections::HashMap;

use super::sat::{Lit, Solver};

/// An AIG literal: `2·node` (the node's value) or `2·node + 1` (its negation). Node 0 is the
/// constant false.
pub type L = u32;

/// Constant false.
pub const FALSE: L = 0;
/// Constant true.
pub const TRUE: L = 1;

#[derive(Copy, Clone, Debug)]
enum Node {
    Const,
    Input,
    And(L, L),
}

/// The graph.
#[derive(Debug)]
pub struct Aig {
    nodes: Vec<Node>,
    hash: HashMap<(L, L), L>,
}

impl Default for Aig {
    fn default() -> Self {
        Aig::new()
    }
}

impl Aig {
    /// An empty graph (the constant only).
    pub fn new() -> Aig {
        Aig {
            nodes: vec![Node::Const],
            hash: HashMap::new(),
        }
    }

    /// The number of nodes (gates, inputs and the constant).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph has only its constant.
    pub fn is_empty(&self) -> bool {
        self.nodes.len() == 1
    }

    /// A new input.
    pub fn input(&mut self) -> L {
        self.nodes.push(Node::Input);
        ((self.nodes.len() - 1) as u32) << 1
    }

    /// `a ∧ b`.
    pub fn and(&mut self, a: L, b: L) -> L {
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        if a == FALSE || a == b ^ 1 {
            return FALSE;
        }
        if a == TRUE || a == b {
            return b;
        }
        if let Some(&g) = self.hash.get(&(a, b)) {
            return g;
        }
        self.nodes.push(Node::And(a, b));
        let g = ((self.nodes.len() - 1) as u32) << 1;
        self.hash.insert((a, b), g);
        g
    }

    /// `a ∨ b`.
    pub fn or(&mut self, a: L, b: L) -> L {
        self.and(a ^ 1, b ^ 1) ^ 1
    }

    /// `a ⊕ b`.
    pub fn xor(&mut self, a: L, b: L) -> L {
        if a == FALSE {
            return b;
        }
        if b == FALSE {
            return a;
        }
        if a == TRUE {
            return b ^ 1;
        }
        if b == TRUE {
            return a ^ 1;
        }
        if a == b {
            return FALSE;
        }
        if a == b ^ 1 {
            return TRUE;
        }
        let p = self.and(a, b ^ 1);
        let q = self.and(a ^ 1, b);
        self.or(p, q)
    }

    /// `a ↔ b`.
    pub fn xnor(&mut self, a: L, b: L) -> L {
        self.xor(a, b) ^ 1
    }

    /// `c ? t : e`.
    pub fn mux(&mut self, c: L, t: L, e: L) -> L {
        if c == TRUE || t == e {
            return t;
        }
        if c == FALSE {
            return e;
        }
        let p = self.and(c, t);
        let q = self.and(c ^ 1, e);
        self.or(p, q)
    }

    /// The majority of three (a full adder's carry).
    pub fn maj(&mut self, a: L, b: L, c: L) -> L {
        // A constant makes it a conjunction or a disjunction of the others, and two operands
        // equal or opposite decide it (comparisons with a constant and additions of one meet
        // these at every bit).
        for (k, x, y) in [(a, b, c), (b, a, c), (c, a, b)] {
            if k == FALSE {
                return self.and(x, y);
            }
            if k == TRUE {
                return self.or(x, y);
            }
            if x == y {
                return x;
            }
            if x == y ^ 1 {
                return k;
            }
        }
        let ab = self.and(a, b);
        let ac = self.and(a, c);
        let bc = self.and(b, c);
        let t = self.or(ab, ac);
        self.or(t, bc)
    }

    /// The value of every node, the `k`-th input (in creation order) taking `input(k)`: nodes
    /// are created after their operands, so one pass in order evaluates them all.
    pub fn eval_all(&self, input: impl Fn(u32) -> bool) -> Vec<bool> {
        let mut vals = Vec::with_capacity(self.nodes.len());
        let mut k = 0;
        for n in &self.nodes {
            let v = match *n {
                Node::Const => false,
                Node::Input => {
                    k += 1;
                    input(k - 1)
                }
                Node::And(a, b) => {
                    let get = |l: L| vals[(l >> 1) as usize] != (l & 1 == 1);
                    get(a) && get(b)
                }
            };
            vals.push(v);
        }
        vals
    }

    /// The value of every node for 64 input assignments at once: bit `j` of input `k`'s word
    /// is its value in assignment `j`.
    pub fn eval_words(&self, input: impl Fn(u32) -> u64) -> Vec<u64> {
        let mut vals: Vec<u64> = Vec::with_capacity(self.nodes.len());
        let mut k = 0;
        for n in &self.nodes {
            let v = match *n {
                Node::Const => 0,
                Node::Input => {
                    k += 1;
                    input(k - 1)
                }
                Node::And(a, b) => {
                    let get = |l: L| {
                        let x = vals[(l >> 1) as usize];
                        if l & 1 == 1 { !x } else { x }
                    };
                    get(a) & get(b)
                }
            };
            vals.push(v);
        }
        vals
    }

    /// A literal's word in `eval_words`'s result.
    pub fn word(vals: &[u64], l: L) -> u64 {
        let x = vals[(l >> 1) as usize];
        if l & 1 == 1 { !x } else { x }
    }

    /// A literal's value in `eval_all`'s result.
    pub fn value(vals: &[bool], l: L) -> bool {
        vals[(l >> 1) as usize] != (l & 1 == 1)
    }

    /// The conjunction of many.
    pub fn and_all(&mut self, ls: &[L]) -> L {
        ls.iter().fold(TRUE, |acc, &l| self.and(acc, l))
    }

    /// The disjunction of many.
    pub fn or_all(&mut self, ls: &[L]) -> L {
        ls.iter().fold(FALSE, |acc, &l| self.or(acc, l))
    }
}

/// The clauses of an AIG's goals for a SAT solver: a variable per node the clauses need
/// (inputs keep theirs for reading models back).
///
/// Gates are recognized before encoding (Tseitin's, with gate detection; see Eén, Mishchenko
/// and Sörensson, "Applying Logic Synthesis for Speeding Up SAT", 2007): an exclusive or or a
/// multiplexer (three and-gates, the inner two used nowhere else) is one variable and four
/// clauses, not three variables and nine; and a tree of and-gates, the inner ones used nowhere
/// else, is one variable and a clause per input plus one. Fewer variables and clauses make
/// every propagation cheaper, and the circuits of adders and multipliers are made of these.
#[derive(Debug)]
pub struct Cnf {
    /// AIG node → solver variable (`NONE` for a node without one).
    var: Vec<u32>,
    /// The clauses, as given to the solver, when kept (for a certificate).
    pub clauses: Vec<Vec<Lit>>,
    keep: bool,
    emitted: usize,
}

const NONE: u32 = u32::MAX;

/// How a node that has a variable is encoded.
enum Gate {
    /// `x ⊕ y`.
    Xor(L, L),
    /// `¬(c ? t : e)`.
    NotMux(L, L, L),
    /// The conjunction of `leaves[start..end]`.
    And(usize, usize),
}

impl Cnf {
    /// Clauses for every gate `roots` reach, added to `solver` (and kept in
    /// [`clauses`](Self::clauses) when `keep`).
    pub fn encode(aig: &Aig, roots: &[L], solver: &mut Solver, keep: bool) -> Cnf {
        let n = aig.nodes.len();
        // Uses of each node within the cone (a root counts as a use, so it keeps its
        // variable): nodes come after their operands, so one pass down from the top counts them.
        let mut uses = vec![0u32; n];
        for &r in roots {
            uses[(r >> 1) as usize] += 2;
        }
        for i in (1..n).rev() {
            if uses[i] > 0
                && let Node::And(a, b) = aig.nodes[i]
            {
                uses[(a >> 1) as usize] += 1;
                uses[(b >> 1) as usize] += 1;
            }
        }
        let inner = |l: L| -> Option<(L, L)> {
            match aig.nodes[(l >> 1) as usize] {
                Node::And(a, b) if uses[(l >> 1) as usize] == 1 => Some((a, b)),
                _ => None,
            }
        };
        // `¬P ∧ ¬Q` with P and Q inner gates: an exclusive or or a multiplexer.
        let gate2 = |a: L, b: L| -> Option<Gate> {
            if a & 1 == 0 || b & 1 == 0 {
                return None;
            }
            let ((p0, p1), (q0, q1)) = (inner(a)?, inner(b)?);
            if (q0 == p0 ^ 1 && q1 == p1 ^ 1) || (q0 == p1 ^ 1 && q1 == p0 ^ 1) {
                // P ∨ Q = (p0 ↔ p1).
                return Some(Gate::Xor(p0, p1));
            }
            for (c, t) in [(p0, p1), (p1, p0)] {
                for (d, e) in [(q0, q1), (q1, q0)] {
                    if d == c ^ 1 {
                        return Some(Gate::NotMux(c, t, e));
                    }
                }
            }
            None
        };
        // Which nodes get a variable, and how each is encoded: from the top down, a node's
        // users are decided before it.
        let mut need = vec![false; n];
        for &r in roots {
            need[(r >> 1) as usize] = true;
        }
        let mut gates: Vec<(u32, Gate)> = Vec::new();
        let mut leaves: Vec<L> = Vec::new();
        let mut stack = Vec::new();
        for i in (1..n).rev() {
            if !need[i] {
                continue;
            }
            let Node::And(a, b) = aig.nodes[i] else {
                continue;
            };
            let gate = gate2(a, b).unwrap_or_else(|| {
                // The inputs of the tree of and-gates under `i`.
                let start = leaves.len();
                stack.extend([b, a]);
                while let Some(l) = stack.pop() {
                    match inner(l) {
                        Some((x, y)) if l & 1 == 0 && gate2(x, y).is_none() => {
                            stack.extend([y, x]);
                        }
                        _ => leaves.push(l),
                    }
                }
                Gate::And(start, leaves.len())
            });
            let operands: &[L] = match &gate {
                Gate::Xor(x, y) => &[*x, *y],
                Gate::NotMux(c, t, e) => &[*c, *t, *e],
                Gate::And(s, e) => &leaves[*s..*e],
            };
            for &l in operands {
                need[(l >> 1) as usize] = true;
            }
            gates.push((i as u32, gate));
        }
        let mut cnf = Cnf {
            var: vec![NONE; n],
            clauses: Vec::new(),
            keep,
            emitted: 0,
        };
        for (i, _) in need.iter().enumerate().filter(|(_, n)| **n) {
            cnf.var[i] = solver.new_var();
        }
        if need[0] {
            let f = cnf.lit(FALSE);
            cnf.push(solver, &[!f]);
        }
        let mut long = Vec::new();
        for (i, gate) in gates.into_iter().rev() {
            let g = Lit::pos(cnf.var[i as usize]);
            match gate {
                Gate::Xor(x, y) => {
                    let (x, y) = (cnf.lit(x), cnf.lit(y));
                    cnf.push(solver, &[!g, x, y]);
                    cnf.push(solver, &[!g, !x, !y]);
                    cnf.push(solver, &[g, !x, y]);
                    cnf.push(solver, &[g, x, !y]);
                }
                Gate::NotMux(c, t, e) => {
                    let (c, t, e) = (cnf.lit(c), cnf.lit(t), cnf.lit(e));
                    // g = ¬(c ? t : e).
                    cnf.push(solver, &[!c, !t, !g]);
                    cnf.push(solver, &[!c, t, g]);
                    cnf.push(solver, &[c, !e, !g]);
                    cnf.push(solver, &[c, e, g]);
                }
                Gate::And(s, e) => {
                    long.clear();
                    long.push(g);
                    for &l in &leaves[s..e] {
                        let x = cnf.lit(l);
                        cnf.push(solver, &[!g, x]);
                        long.push(!x);
                    }
                    cnf.push(solver, &long);
                }
            }
        }
        cnf
    }

    /// The number of clauses given to the solver.
    pub fn emitted(&self) -> usize {
        self.emitted
    }

    fn push(&mut self, solver: &mut Solver, c: &[Lit]) {
        self.emitted += 1;
        if self.keep {
            self.clauses.push(c.to_vec());
        }
        solver.add_clause(c);
    }

    /// The solver literal of an AIG literal (its node must have a variable: a root's does).
    pub fn lit(&self, l: L) -> Lit {
        let v = self.var[(l >> 1) as usize];
        debug_assert_ne!(v, NONE, "a node without a variable");
        Lit::new(v, l & 1 == 0)
    }

    /// Asserts an AIG literal (a root's).
    pub fn assert(&mut self, l: L, solver: &mut Solver) {
        let x = self.lit(l);
        self.push(solver, &[x]);
    }

    /// The value of an AIG literal in a model: an input's, or a root's (inputs not reached
    /// are false).
    pub fn value(&self, l: L, model: &[bool]) -> bool {
        let v = match self.var.get((l >> 1) as usize) {
            Some(&v) if v != NONE => model.get(v as usize).copied().unwrap_or(false),
            _ => false,
        };
        v != (l & 1 == 1)
    }
}
