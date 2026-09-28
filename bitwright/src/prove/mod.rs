//! A native prover (feature `prove`): expressions are bit-blasted into an and-inverter graph,
//! encoded as clauses, and decided by bitwright's own CDCL SAT solver. A proof of validity comes
//! with a certificate, the clauses and a DRUP proof of their unsatisfiability, that an
//! independent checker ([`Certificate::check`]) verifies, and that any DRAT checker can read
//! ([`Certificate::dimacs`], [`Certificate::drat`]). A refutation comes with a counterexample:
//! a value for every symbol, checked by bitwright's evaluator before it is returned. A question
//! the search does not settle within its budget says why ([`Unknown`]), and a [`Question`]
//! goes on from where the search stopped, under a larger budget, without starting over.
//!
//! No other solver is involved. Bit-vector operators are blasted with bitwright's total
//! semantics (SMT-LIB's); floating-point operators with IEEE 754's and bitwright's canonical
//! NaN ([`crate::fp`]); extension operations through their [`expand`](crate::ext::ExtOp::expand)
//! definition, if they have one (otherwise the question is not decided).
//!
//! ```
//! use bitwright::prove::{Config, Outcome, equal};
//! use bitwright::{Context, ParseOptions, Width};
//!
//! let mut cx = Context::new();
//! let o = ParseOptions::width(Width::W32);
//! let (a, b) = (cx.parse("(x ^ y) + 2 * (x & y)", &o)?, cx.parse("x + y", &o)?);
//! let Outcome::Proved(Some(cert)) = equal(&mut cx, a, b, &Config::default().with_certificate(true))? else {
//!     panic!("not proved")
//! };
//! assert!(cert.check().is_ok());
//! let c = cx.parse("x | y", &o)?;
//! assert!(matches!(equal(&mut cx, a, c, &Config::default())?, Outcome::Refuted(_)));
//! # Ok::<(), bitwright::Error>(())
//! ```

pub mod aig;
pub mod blast;
pub mod drup;
mod fp;
mod rule;
pub mod sat;
#[cfg(test)]
mod tests;

use crate::error::Error;
use crate::expr::{Context, Expr, OpCode};
use crate::facts::Assumptions;
use crate::hash::IdMap;
use crate::{BitVec, SymbolKey, Width};

pub use rule::{RuleOutcome, WidthReport, rule, rule_all_widths};

use aig::{Aig, Cnf, L};
use blast::Bits;
pub use sat::Limits;
use sat::{Answer, Lit, Solver, Step};

/// How hard to try, and what to keep.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Config {
    /// The most conflicts the SAT solver may meet (then [`Unknown::Budget`]).
    pub max_conflicts: u64,
    /// The most propagations the SAT solver may do (then [`Unknown::Budget`]; `u64::MAX`, the
    /// default: no limit). A conflict costs more propagation on a larger circuit, so this bounds
    /// the time a question takes more closely than conflicts do, and as deterministically.
    pub max_propagations: u64,
    /// The most AIG nodes the question may take (then [`Unknown::TooLarge`]).
    pub max_nodes: usize,
    /// Keep a [`Certificate`] for a proof.
    pub certificate: bool,
    /// Simplify the question with bitwright's own engine before blasting it (the
    /// deobfuscation strategy, with the MBA service when the `mba` feature is on): identities
    /// the engine settles at the word level (products of MBA, say, which bit-level SAT finds
    /// hard) are answered at once, and the rest is blasted smaller. Not with a certificate: a
    /// certificate is of the question as asked.
    pub simplify: bool,
    /// Assignments of the symbols to evaluate on the circuit before the search, 64 at a time
    /// (from a fixed seed; the first two all zeros and all ones): a refutation that a fraction of
    /// the values give is found without the search, which needs many conflicts for it on a
    /// large circuit. 0: none.
    pub samples: u32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            max_conflicts: 1_000_000,
            max_propagations: u64::MAX,
            max_nodes: 4_000_000,
            certificate: false,
            simplify: true,
            samples: 256,
        }
    }
}

setters!(Config {
    with_max_conflicts: max_conflicts: u64,
    with_max_propagations: max_propagations: u64,
    with_max_nodes: max_nodes: usize,
    with_certificate: certificate: bool,
    with_simplify: simplify: bool,
    with_samples: samples: u32,
});

impl Config {
    /// The search's limits: [`max_conflicts`](Self::max_conflicts) and
    /// [`max_propagations`](Self::max_propagations).
    pub fn limits(&self) -> Limits {
        Limits {
            conflicts: self.max_conflicts,
            propagations: self.max_propagations,
        }
    }
}

/// The engine that simplifies questions (see [`Config::simplify`]).
fn simplifier() -> &'static crate::engine::Engine {
    static ENGINE: std::sync::OnceLock<crate::engine::Engine> = std::sync::OnceLock::new();
    ENGINE.get_or_init(|| {
        #[cfg(feature = "mba")]
        let strategy =
            crate::engine::Strategy::deobfuscate().with_mba(crate::mba::MbaConfig::default());
        #[cfg(not(feature = "mba"))]
        let strategy = crate::engine::Strategy::deobfuscate();
        crate::engine::Engine::builder()
            .builtin()
            .strategy(strategy)
            .build()
            .unwrap_or_else(|_| crate::engine::Engine::standard())
    })
}

/// Clauses and a DRUP proof that they are unsatisfiable.
#[derive(Clone, Debug)]
pub struct Certificate {
    /// The number of variables.
    pub vars: u32,
    /// The clauses.
    pub clauses: Vec<Vec<Lit>>,
    /// The proof.
    pub proof: Vec<Step>,
}

impl Certificate {
    /// Checks the proof with the independent checker.
    pub fn check(&self) -> Result<(), String> {
        drup::check(&self.clauses, &self.proof)
    }

    /// The clauses in DIMACS CNF.
    pub fn dimacs(&self) -> String {
        let mut s = format!("p cnf {} {}\n", self.vars, self.clauses.len());
        for c in &self.clauses {
            for l in c {
                s.push_str(&l.dimacs().to_string());
                s.push(' ');
            }
            s.push_str("0\n");
        }
        s
    }

    /// The proof in DRAT's text format (additions, and deletions prefixed `d`).
    pub fn drat(&self) -> String {
        let mut s = String::new();
        for step in &self.proof {
            let (pre, c) = match step {
                Step::Add(c) => ("", c),
                Step::Delete(c) => ("d ", c),
            };
            s.push_str(pre);
            for l in c {
                s.push_str(&l.dimacs().to_string());
                s.push(' ');
            }
            s.push_str("0\n");
        }
        s
    }
}

/// The answer to a question.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Outcome {
    /// It holds for every value of the symbols (with a certificate, if asked for).
    Proved(Option<Certificate>),
    /// It fails at these values of the symbols (checked by evaluation).
    Refuted(Vec<(SymbolKey, BitVec)>),
    /// Not decided: why.
    Unknown(Unknown),
}

/// Why a question is not decided.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Unknown {
    /// The search reached a limit ([`Config::max_conflicts`] or
    /// [`Config::max_propagations`]) after this much work on the question: more may decide
    /// it, and a [`Question`] goes on from where it stopped.
    Budget {
        /// Conflicts met on the question.
        conflicts: u64,
        /// Propagations done on the question.
        propagations: u64,
    },
    /// The circuit passed [`Config::max_nodes`]: `nodes` had been built when blasting stopped.
    /// The same under any search budget with that cap.
    TooLarge {
        /// AIG nodes built.
        nodes: usize,
    },
    /// Something the prover does not reason about (an extension operation without an
    /// expansion, say). The same under any budget.
    Unsupported(String),
}

impl Unknown {
    /// Whether a larger search budget may decide the question ([`Unknown::Budget`]).
    pub fn is_budget(&self) -> bool {
        matches!(self, Unknown::Budget { .. })
    }
}

impl core::fmt::Display for Unknown {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Unknown::Budget {
                conflicts,
                propagations,
            } => write!(
                f,
                "no answer within {conflicts} conflicts ({propagations} propagations)"
            ),
            Unknown::TooLarge { nodes } => {
                write!(f, "the circuit is too large ({nodes} nodes built)")
            }
            Unknown::Unsupported(why) => f.write_str(why),
        }
    }
}

/// The work a question took, so far.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Stats {
    /// AIG nodes of the circuit (inputs and gates).
    pub nodes: usize,
    /// Assignments evaluated on the circuit ([`Config::samples`]).
    pub samples: u64,
    /// Variables of the clauses.
    pub vars: u32,
    /// Clauses of the question (not learned ones).
    pub clauses: usize,
    /// Learned clauses kept.
    pub learned: usize,
    /// Conflicts.
    pub conflicts: u64,
    /// Decisions.
    pub decisions: u64,
    /// Propagations.
    pub propagations: u64,
}

/// The bits of expressions, blasted on demand.
pub(crate) struct Blaster<'c> {
    pub(crate) cx: &'c mut Context,
    pub(crate) g: Aig,
    bits: IdMap<u32, Bits>,
    /// Each symbol met, with its bits.
    pub(crate) symbols: Vec<(u32, Bits)>,
    max_nodes: usize,
    /// Blasting stopped at `max_nodes`.
    too_large: bool,
}

impl<'c> Blaster<'c> {
    pub(crate) fn new(cx: &'c mut Context, max_nodes: usize) -> Blaster<'c> {
        Blaster {
            cx,
            g: Aig::new(),
            bits: IdMap::default(),
            symbols: Vec::new(),
            max_nodes,
            too_large: false,
        }
    }

    /// The bits of node `root`.
    pub(crate) fn blast(&mut self, root: u32) -> Result<Bits, Error> {
        let order = self.cx.post_order_ids(&[root]);
        for i in order {
            if self.bits.contains_key(&i) {
                continue;
            }
            if self.g.len() > self.max_nodes {
                self.too_large = true;
                return Err(Error::Unsupported("the circuit is too large".into()));
            }
            let b = self.node(i)?;
            self.bits.insert(i, b);
        }
        Ok(self.bits[&root].clone())
    }

    fn node(&mut self, i: u32) -> Result<Bits, Error> {
        use blast::*;
        let n = self.cx.node(i);
        let w = usize::from(n.width);
        let get = |s: &Self, j: u32| s.bits[&j].clone();
        let g = &mut self.g;
        Ok(match n.op {
            OpCode::Const => konst(&self.cx.const_val(i).unwrap_or(BitVec::zero(Width::W1))),
            OpCode::Sym => {
                let b: Bits = (0..w).map(|_| g.input()).collect();
                self.symbols.push((i, b.clone()));
                b
            }
            OpCode::Not => not(g, &self.bits[&n.a]),
            OpCode::Neg => {
                let a = get(self, n.a);
                neg(&mut self.g, &a)
            }
            OpCode::Popcnt => popcnt(g, &self.bits[&n.a]),
            OpCode::Clz => clz(g, &self.bits[&n.a]),
            OpCode::Ctz => ctz(g, &self.bits[&n.a]),
            OpCode::Bswap => bswap(&self.bits[&n.a]),
            OpCode::BitRev => bitrev(&self.bits[&n.a]),
            OpCode::Zext => {
                let mut a = get(self, n.a);
                a.resize(w, aig::FALSE);
                a
            }
            OpCode::Sext => {
                let mut a = get(self, n.a);
                let top = *a.last().unwrap_or(&aig::FALSE);
                a.resize(w, top);
                a
            }
            OpCode::Extract => {
                let a = get(self, n.a);
                let lo = n.b as usize;
                a[lo..lo + w].to_vec()
            }
            OpCode::Concat => {
                let (hi, mut lo) = (get(self, n.a), get(self, n.b));
                lo.extend(hi);
                lo
            }
            OpCode::Select => {
                let (c, t, e) = (get(self, n.a)[0], get(self, n.b), get(self, n.c));
                mux(&mut self.g, c, &t, &e)
            }
            OpCode::Eq | OpCode::Ne | OpCode::Ult | OpCode::Ule | OpCode::Slt | OpCode::Sle => {
                let (a, b) = (get(self, n.a), get(self, n.b));
                let g = &mut self.g;
                let r = match n.op {
                    OpCode::Eq => eq(g, &a, &b),
                    OpCode::Ne => eq(g, &a, &b) ^ 1,
                    OpCode::Ult => ult(g, &a, &b),
                    OpCode::Ule => ule(g, &a, &b),
                    OpCode::Slt => slt(g, &a, &b),
                    _ => sle(g, &a, &b),
                };
                vec![r]
            }
            op if op.as_bin().is_some() => {
                let (a, b) = (get(self, n.a), get(self, n.b));
                let g = &mut self.g;
                match op {
                    OpCode::Add => add(g, &a, &b),
                    OpCode::Sub => sub(g, &a, &b),
                    OpCode::Mul => mul(g, &a, &b),
                    OpCode::UMulHi => mulhi(g, &a, &b, false),
                    OpCode::SMulHi => mulhi(g, &a, &b, true),
                    OpCode::UDiv => udivrem(g, &a, &b).0,
                    OpCode::URem => udivrem(g, &a, &b).1,
                    OpCode::SDiv => sdiv(g, &a, &b),
                    OpCode::SRem => srem(g, &a, &b),
                    OpCode::And => and(g, &a, &b),
                    OpCode::Or => or(g, &a, &b),
                    OpCode::Xor => xor(g, &a, &b),
                    OpCode::Shl => shift(g, &a, &b, true, aig::FALSE),
                    OpCode::LShr => shift(g, &a, &b, false, aig::FALSE),
                    OpCode::AShr => {
                        let s = *a.last().unwrap_or(&aig::FALSE);
                        shift(g, &a, &b, false, s)
                    }
                    OpCode::RotL => rotate(g, &a, &b, true),
                    OpCode::RotR => rotate(g, &a, &b, false),
                    OpCode::Pdep => pdep(g, &a, &b),
                    _ => pext(g, &a, &b),
                }
            }
            op if op.as_ext().is_some() => self.ext(i)?,
            _ => {
                let kids: Vec<Bits> = n.children().map(|c| self.bits[&c].clone()).collect();
                fp::blast(self, i, &kids)?
            }
        })
    }

    /// An extension output, through the operation's expansion.
    fn ext(&mut self, i: u32) -> Result<Bits, Error> {
        let n = self.cx.node(i);
        let Some((_, k)) = n.op.as_ext() else {
            return Err(Error::Unsupported("not an extension output".into()));
        };
        let args: Vec<Expr> = n.children().map(|c| self.cx.handle(c)).collect();
        let reg = self
            .cx
            .registry
            .clone()
            .ok_or_else(|| Error::Unsupported("an extension without its registry".into()))?;
        let op = reg
            .op_at(n.aux)
            .ok_or_else(|| Error::Unsupported("an extension without its operation".into()))?;
        let Some(ex) = op.expand(self.cx, &args) else {
            return Err(Error::Unsupported(format!(
                "`{}` has no expansion to reason about",
                op.name()
            )));
        };
        let outs = ex?;
        let e = self.cx.id(outs[k])?;
        self.blast(e)
    }

    /// Why blasting failed with `e`: [`Unknown::TooLarge`] past `max_nodes`, else the reason
    /// of an unsupported operation. Other errors are errors.
    fn unknown(&self, e: Error) -> Result<Unknown, Error> {
        match e {
            _ if self.too_large => Ok(Unknown::TooLarge {
                nodes: self.g.len(),
            }),
            Error::Unsupported(why) => Ok(Unknown::Unsupported(why)),
            e => Err(e),
        }
    }
}

/// Where a SAT-solved question stands.
struct Search {
    solver: Solver,
    cnf: Cnf,
}

/// A question, blasted and encoded once, then decided in as many steps as its caller likes: a
/// search that ends on its limits stops where it is, keeping what it learned, and the next
/// [`solve`](Self::solve) goes on from there. Asking under 5,000 conflicts, then 15,000 more,
/// costs what asking under 20,000 once does, and answers the same.
///
/// ```
/// use bitwright::prove::{Config, Limits, Outcome, Question};
/// use bitwright::{Context, ParseOptions, Width};
///
/// let mut cx = Context::new();
/// let o = ParseOptions::width(Width::W16);
/// let p = cx.parse("x * x != 0x1234", &o)?;
/// let mut q = Question::valid(&mut cx, p, &Config::default().with_samples(0))?;
/// let mut out = q.solve(&mut cx, Limits::conflicts(10))?;
/// while let Outcome::Unknown(why) = &out {
///     assert!(why.is_budget());
///     out = q.solve(&mut cx, Limits::conflicts(10))?;
/// }
/// assert!(matches!(out, Outcome::Proved(_))); // 0x1234 is not a square mod 2^16
/// assert!(q.stats().conflicts > 0);
/// # Ok::<(), bitwright::Error>(())
/// ```
pub struct Question {
    /// The question as asked (with the constraints, as `¬(c1 ∧ …) ∨ p`), checked by evaluation.
    root: Expr,
    /// Each symbol of the question with its bits (none for one the circuit does not read,
    /// which takes 0).
    symbols: Vec<(SymbolKey, Width, Bits)>,
    search: Option<Box<Search>>,
    outcome: Option<Outcome>,
    stats: Stats,
}

impl core::fmt::Debug for Question {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Question")
            .field("outcome", &self.outcome)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

impl Question {
    /// Whether the 1-bit `p` is true for every value of its symbols.
    pub fn valid(cx: &mut Context, p: Expr, cfg: &Config) -> Result<Question, Error> {
        Question::valid_under(cx, p, None, cfg)
    }

    /// Whether `a` and `b` (of one width) are equal for every value of their symbols.
    pub fn equal(cx: &mut Context, a: Expr, b: Expr, cfg: &Config) -> Result<Question, Error> {
        let (ai, bi) = (cx.id(a)?, cx.id(b)?);
        if cx.wid(ai) != cx.wid(bi) {
            return Err(crate::WidthError::Mismatch {
                left: cx.wid(ai),
                right: cx.wid(bi),
            }
            .into());
        }
        let p = cx.c_cmp(crate::CmpOp::Eq, ai, bi)?;
        let p = cx.handle(p);
        Question::valid(cx, p, cfg)
    }

    /// Whether `p` is true wherever the constraints of `assumptions` hold. The question is
    /// simplified (see [`Config::simplify`]), blasted, sampled (see [`Config::samples`]) and
    /// encoded here; each of these may decide it.
    pub fn valid_under(
        cx: &mut Context,
        p: Expr,
        assumptions: Option<&Assumptions>,
        cfg: &Config,
    ) -> Result<Question, Error> {
        let pi = cx.id(p)?;
        if cx.wid(pi) != 1 {
            return Err(crate::WidthError::Mismatch {
                left: 1,
                right: cx.wid(pi),
            }
            .into());
        }
        let mut q = Question {
            root: p,
            symbols: Vec::new(),
            search: None,
            outcome: None,
            stats: Stats::default(),
        };
        let decided = |mut q: Question, o: Outcome| {
            q.outcome = Some(o);
            Ok(q)
        };
        // The constraints as 1-bit expressions: each predicate holds, or does not.
        let mut cons: Vec<u32> = Vec::new();
        if let Some(a) = assumptions {
            for (_, e, f) in a.constraints() {
                let i = cx.id(e)?;
                if let Some(v) = f.as_constant() {
                    let k = cx.mk_const(&v)?;
                    cons.push(cx.c_cmp(crate::CmpOp::Eq, i, k)?);
                } else {
                    let why = "a constraint that is not a predicate's value".into();
                    return decided(q, Outcome::Unknown(Unknown::Unsupported(why)));
                }
            }
        }
        // The root checked by evaluation: `p` where the constraints hold, as `¬(c1 ∧ …) ∨ p`.
        let mut root = pi;
        for &c in &cons {
            let nc = cx.c_un(crate::UnOp::Not, c)?;
            root = cx.c_bin(crate::BinOp::Or, nc, root)?;
        }
        q.root = cx.handle(root);
        // The question simplified (not for a certificate, which must be of the question itself).
        let mut goal_node = pi;
        if cfg.simplify && !cfg.certificate {
            let run = match assumptions {
                Some(a) => crate::engine::Run::default().with_assumptions(a),
                None => crate::engine::Run::default(),
            };
            let out = simplifier().run(cx, &[p], run)?.roots[0];
            let s = cx.id(out.expr)?;
            if cx.const_val(s).is_some_and(|v| !v.is_zero()) {
                return decided(q, Outcome::Proved(None));
            }
            // A result that relies on constraints holds where they do, which is all a proof
            // under them needs.
            goal_node = s;
        }
        let mut b = Blaster::new(cx, cfg.max_nodes);
        let mut blasted = || -> Result<(L, Vec<L>), Error> {
            let goal = b.blast(goal_node)?[0];
            let mut extra = Vec::new();
            for &c in &cons {
                extra.push(b.blast(c)?[0]);
            }
            Ok((goal, extra))
        };
        let (goal, extra) = match blasted() {
            Ok(r) => r,
            Err(e) => {
                let why = b.unknown(e)?;
                return decided(q, Outcome::Unknown(why));
            }
        };
        q.stats.nodes = b.g.len();
        if b.g.len() > cfg.max_nodes {
            let why = Unknown::TooLarge { nodes: b.g.len() };
            return decided(q, Outcome::Unknown(why));
        }
        // Every symbol of the question, with its bits if the circuit reads it.
        for (s, bits) in &b.symbols {
            let h = b.cx.handle(*s);
            let key =
                b.cx.symbol_id(h)?
                    .and_then(|id| b.cx.symbol_key(id))
                    .cloned()
                    .ok_or_else(|| Error::Contract("a symbol without its key".into()))?;
            q.symbols.push((key, b.cx.width_of(*s), bits.clone()));
        }
        for id in b.cx.symbols_in(&[q.root])? {
            if let (Some(k), Some(w)) = (b.cx.symbol_key(id).cloned(), b.cx.symbol_width(id))
                && !q.symbols.iter().any(|(kk, _, _)| *kk == k)
            {
                q.symbols.push((k, w, Vec::new()));
            }
        }
        // Sampled: a lane where the constraints hold and the goal does not refutes it.
        let batches = cfg.samples.div_ceil(64);
        for batch in 0..batches {
            let vals = b.g.eval_words(|k| sample(batch, k));
            q.stats.samples += 64;
            let mut bad = !Aig::word(&vals, goal);
            for &e in &extra {
                bad &= Aig::word(&vals, e);
            }
            if bad != 0 {
                let lane = bad.trailing_zeros();
                let ce = Question::counterexample(&q.symbols, |l| {
                    (Aig::word(&vals, l) >> lane) & 1 == 1
                });
                let o = q.refuted(b.cx, ce)?;
                return decided(q, o);
            }
        }
        let mut solver = Solver::new();
        if cfg.certificate {
            solver.log_proof();
        }
        let mut roots = vec![goal];
        roots.extend_from_slice(&extra);
        let mut cnf = Cnf::encode(&b.g, &roots, &mut solver, cfg.certificate);
        for &e in &extra {
            cnf.assert(e, &mut solver);
        }
        cnf.assert(goal ^ 1, &mut solver);
        q.stats.vars = solver.num_vars();
        q.stats.clauses = cnf.emitted();
        q.search = Some(Box::new(Search { solver, cnf }));
        Ok(q)
    }

    /// Searches within `limits` more work (counted from this call): the answer, or
    /// [`Unknown::Budget`] with the work done on the question so far, and the next call goes on
    /// from there. Once decided (here or when built), the answer is kept and returned again.
    /// `cx` is the context the question was built in.
    pub fn solve(&mut self, cx: &mut Context, limits: Limits) -> Result<Outcome, Error> {
        if let Some(o) = &self.outcome {
            return Ok(o.clone());
        }
        let Some(s) = self.search.as_mut() else {
            return Err(Error::Contract(
                "a question neither decided nor open".into(),
            ));
        };
        let answer = s.solver.solve_within(limits);
        self.stats.learned = s.solver.num_learnts();
        self.stats.conflicts = s.solver.conflicts;
        self.stats.decisions = s.solver.decisions;
        self.stats.propagations = s.solver.propagations;
        let o = match answer {
            Answer::Unknown => {
                return Ok(Outcome::Unknown(Unknown::Budget {
                    conflicts: s.solver.conflicts,
                    propagations: s.solver.propagations,
                }));
            }
            Answer::Unsat => {
                let cert = s.solver.take_proof().map(|proof| Certificate {
                    vars: s.solver.num_vars(),
                    clauses: core::mem::take(&mut s.cnf.clauses),
                    proof,
                });
                Outcome::Proved(cert)
            }
            Answer::Sat(model) => {
                let cnf = &s.cnf;
                let ce = Question::counterexample(&self.symbols, |l| cnf.value(l, &model));
                self.refuted(cx, ce)?
            }
        };
        self.search = None;
        self.outcome = Some(o.clone());
        Ok(o)
    }

    /// The answer, once decided.
    pub fn outcome(&self) -> Option<&Outcome> {
        self.outcome.as_ref()
    }

    /// The work done on the question so far.
    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// The counterexample an assignment of the circuit's inputs gives.
    fn counterexample(symbols: &[(SymbolKey, Width, Bits)], bit: impl Fn(L) -> bool) -> Model {
        symbols
            .iter()
            .map(|(k, w, bits)| {
                let mut limbs = [0u64; 8];
                for (j, &b) in bits.iter().enumerate() {
                    if bit(b) {
                        limbs[j / 64] |= 1 << (j % 64);
                    }
                }
                (k.clone(), BitVec::wrapping_from_limbs(*w, &limbs))
            })
            .collect()
    }

    /// `Refuted(ce)`, checked by bitwright's evaluator: a solver or blaster bug cannot produce a
    /// false refutation unnoticed.
    fn refuted(&self, cx: &mut Context, ce: Model) -> Result<Outcome, Error> {
        let env =
            crate::FnEnv(|k: &SymbolKey, _| ce.iter().find(|(kk, _)| kk == k).map(|(_, v)| *v));
        let v = cx.eval(&[self.root], &env)?[0];
        if v.is_zero() {
            Ok(Outcome::Refuted(ce))
        } else {
            Err(Error::Contract(format!(
                "the prover's counterexample does not refute {}",
                cx.display(self.root)
            )))
        }
    }
}

/// Input `k`'s word in sample batch `batch` (splitmix64 of the pair); batch 0's lanes 0 and 1
/// are all zeros and all ones.
fn sample(batch: u32, k: u32) -> u64 {
    let mut z = (u64::from(batch) << 32 | u64::from(k)).wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    if batch == 0 { (z & !0b11) | 0b10 } else { z }
}

/// Whether the 1-bit `p` is true for every value of its symbols.
pub fn valid(cx: &mut Context, p: Expr, cfg: &Config) -> Result<Outcome, Error> {
    valid_under(cx, p, None, cfg)
}

/// Whether `p` is true wherever the constraints of `assumptions` hold.
pub fn valid_under(
    cx: &mut Context,
    p: Expr,
    assumptions: Option<&Assumptions>,
    cfg: &Config,
) -> Result<Outcome, Error> {
    Question::valid_under(cx, p, assumptions, cfg)?.solve(cx, cfg.limits())
}

/// Whether `a` and `b` (of one width) are equal for every value of their symbols.
pub fn equal(cx: &mut Context, a: Expr, b: Expr, cfg: &Config) -> Result<Outcome, Error> {
    Question::equal(cx, a, b, cfg)?.solve(cx, cfg.limits())
}

/// Values of symbols, as a counterexample or a model gives them.
pub type Model = Vec<(SymbolKey, BitVec)>;

/// A value of the symbols that makes the 1-bit `p` true, if there is one: `Refuted` of `¬p`
/// read back as a model (`Ok(Some)`), `Ok(None)` when `p` is unsatisfiable, and `Err` with the
/// reason when not decided.
pub fn satisfy(
    cx: &mut Context,
    p: Expr,
    cfg: &Config,
) -> Result<Result<Option<Model>, Unknown>, Error> {
    let np = cx.un(crate::UnOp::Not, p)?;
    Ok(match valid(cx, np, cfg)? {
        Outcome::Proved(_) => Ok(None),
        Outcome::Refuted(m) => Ok(Some(m)),
        Outcome::Unknown(why) => Err(why),
    })
}
