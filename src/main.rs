extern crate rand;
extern crate rand_distr;

use rand::prelude::*;
use rand_distr::{Binomial, Normal};

const C: usize = 15;
const N: usize = 300;
const T: usize = 50;
const K: usize = 8;
const P: [f64;8] = [0.001,0.02,0.003,0.03,0.005,0.001,0.003,0.008];
const PX: [f64;8] = [1000.0, 3000.0, 5000.0, 10000.0, 15000.0, 12500.0, 18000.0, 12000.0];
const PROB: f64 = 0.4;
const MU_MIN: f64 = 0.0;
const MU_MAX: f64 = 10.0;
const P_DEFAULT: u64 = 100;
const P_NORMAL: f64 = 10000000.0;
const VALUE: f64 = 250000.0;
const GOVERNMENT_P: f64 = P_NORMAL*(P_DEFAULT as f64)*(P_DEFAULT as f64);

#[derive(Clone, Debug)]
struct Model {
    mu: Vec<Vec<Vec<f64>>>,
    u: Vec<Vec<f64>>,
    p: Vec<Vec<f64>>,
    coff: Vec<Vec<f64>>,
    adj: Vec<Vec<Vec<usize>>>,
    q: f64,
}

impl Model {
    fn new() -> Self {
        let mut mu = vec![vec![vec![0.0;T+1];K];N];
        let mut u = vec![vec![0.0;T+1];N];
        let mut p = vec![vec![0.0;T+1];N];
        let mut rng = thread_rng();
        for i in 0..N {
            for j in 0..K {
                mu[i][j][0] = rng.gen_range(MU_MIN..=MU_MAX);
            }
        }
        let dist = Binomial::new(P_DEFAULT, 0.4).unwrap();
        let dist2 = Normal::new(0.0, 1.0).unwrap();
        for i in 1..N {
            p[i][0] = (rng.sample(&dist) as f64)*P_NORMAL;
        }
        p[0][0] = GOVERNMENT_P;
        for i in 1..N {
            u[i][0] = (rng.sample(&dist2) as f64).exp();
        }
        let mut adj: Vec<Vec<Vec<usize>>> = vec![vec![vec![];K];N];
        for i in 0..N {
            for c in 0..K {
                for j in 0..N {
                    let val = rng.gen_range(0.0..=1.0);
                    if val >= 0.8 {
                        adj[i][c].push(j);
                    }
                }
            }
        }
        let q = rng.gen_range(0.0..=1.0 / (C as f64).ln());
        let mut sum = 0.0;
        for k in 1..=C {
            sum += 1.0 / (C as f64) * (1.0 / (k as f64));
        }
        let mut coff: Vec<Vec<f64>> = vec![vec![0.0;K];N];
        sum *= q;
        for i in 0..N {
            for j in 0..K {
                coff[i][j] = (adj[i][j].len() as f64)*sum;
            }
        }
        Self {
            mu,
            u,
            p,
            coff,
            adj,
            q,
        }
    }

    fn run_model_1(&mut self, time: usize) -> f64 {
        let mut p = self.p.clone();
        let mut mu = self.mu.clone();
        let mut u = self.u.clone();
        for i in 1..N {
            p[i][0] += VALUE;
            p[0][0] -= VALUE;
        }
        for t in 1..=time {
            p[0][t] = p[0][t-1];
            for a in 0..N {
                for x in 0..K {
                    u[a][t] = u[a][t-1];
                    p[a][t] = p[a][t-1];
                    mu[a][x][t] = self.coff[a][x] * mu[a][x][t-1];
                }
            }
            for a in 0..N {
                for x in 0..K {
                    for b in self.adj[a][x].clone() {
                        if p[a][t-1] >= PX[x] {
                            u[a][t] += self.q*mu[a][x][t-1];
                            u[b][t] -= self.q*mu[a][x][t-1];
                            p[a][t] -= self.q*PX[x];
                            p[b][t] += self.q*(1.0 - P[x])*PX[x];
                            p[0][t] += self.q*P[x]*PX[x];
                        }
                    } 
                }
            }
        }
        let mut w = u[1][time];
        for i in 1..N {
            if w > u[i][time] {
                w = u[i][time];
            }
        }
        w
    }

    fn run_model_2(&mut self, time: usize) -> f64 {
        let mut p = self.p.clone();
        let mut mu = self.mu.clone();
        let mut u = self.u.clone();
        for t in 1..=time {
            p[0][t] = p[0][t-1];
            for a in 0..N {
                for x in 0..K {
                    u[a][t] = u[a][t-1];
                    p[a][t] = p[a][t-1];
                    mu[a][x][t] = self.coff[a][x] * mu[a][x][t-1];
                }
            }
            for a in 0..N {
                for x in 0..K {
                    for b in self.adj[a][x].clone() {
                        if p[a][t-1] >= PX[x] {
                            u[a][t] += self.q*mu[a][x][t-1];
                            u[b][t] -= self.q*mu[a][x][t-1];
                            p[a][t] -= self.q*PX[x];
                            p[b][t] += self.q*(1.0 - P[x])*PX[x];
                            p[0][t] += self.q*P[x]*PX[x];
                        }
                    } 
                }
            }
        }
        let mut w = u[1][time];
        for i in 1..N {
            if w > u[i][time] {
                w = u[i][time];
            }
        }
        w
    }

    fn run_model(&mut self, time: usize) -> f64 {
        self.run_model_1(time) / self.run_model_2(time)
    }
}

fn main() {
    let mut model = Model::new();
    println!("{}", model.run_model(T));
}
