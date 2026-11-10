use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::os::unix::fs::PermissionsExt;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};
use wgt::{
    Client, InstanceOptions, Loss, ModelDefinition, Optimizer, Targets, TrainOptions,
    weights::{Tensor, WeightSet},
};

const FAMILIES: [&str; 4] = ["mlp", "cnn", "resnet", "transformer"];
const BASELINES: [&str; 3] = ["normal", "xavier_uniform", "kaiming_normal"];
const REVISION: &str = include_str!("../WEIGHTLANG_REVISION");
type Weights = BTreeMap<String, Vec<f32>>;

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Config {
    version: u32,
    architecture_seed: u64,
    draws: usize,
    initialization_seeds: Vec<u64>,
    train_seed: u64,
    test_seed: u64,
    train_samples: usize,
    test_samples: usize,
    steps: usize,
    batch_size: usize,
    learning_rate: f32,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Submission {
    version: u32,
    trials: BTreeMap<String, Weights>,
}
struct Trial {
    id: String,
    task: String,
    family: String,
    seed: u64,
    definition: ModelDefinition,
}

// SplitMix64, including the integer-to-uniform conversion, is part of protocol v1.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / 9007199254740992.0
    }
    fn index(&mut self, n: usize) -> usize {
        let threshold = (n as u64).wrapping_neg() % n as u64;
        loop {
            let x = self.next();
            if x >= threshold {
                return (x % n as u64) as usize;
            }
        }
    }
    fn normal(&mut self) -> f64 {
        (-2.0 * self.unit().ln()).sqrt() * (std::f64::consts::TAU * self.unit()).cos()
    }
}
fn stream(seed: u64, name: &str) -> Rng {
    let hash = Sha256::digest([seed.to_le_bytes().as_slice(), name.as_bytes()].concat());
    Rng(u64::from_le_bytes(hash[..8].try_into().unwrap()))
}
fn config(root: &Path) -> Result<Config> {
    let c: Config = serde_json::from_slice(&fs::read(root.join("config/benchmark.json"))?)?;
    ensure!(
        c.version == 1 && (1..=32).contains(&c.draws),
        "unsupported config"
    );
    ensure!(
        !c.initialization_seeds.is_empty() && c.initialization_seeds.len() <= 10,
        "invalid seeds"
    );
    let mut seeds = c.initialization_seeds.clone();
    seeds.sort();
    seeds.dedup();
    ensure!(
        seeds.len() == c.initialization_seeds.len(),
        "duplicate seeds"
    );
    ensure!(
        c.train_seed != c.test_seed,
        "data splits must use independent seeds"
    );
    ensure!(
        (3..=4096).contains(&c.train_samples) && (3..=4096).contains(&c.test_samples),
        "invalid dataset size"
    );
    ensure!(
        c.steps > 0 && c.steps <= 1000 && c.batch_size > 0 && c.batch_size <= c.train_samples,
        "invalid budget"
    );
    ensure!(
        c.learning_rate.is_finite() && c.learning_rate > 0.0,
        "invalid rate"
    );
    Ok(c)
}
fn trials(root: &Path, c: &Config) -> Result<Vec<Trial>> {
    let mut rng = stream(c.architecture_seed, "architectures");
    let mut result = Vec::new();
    let mut pool = Vec::new();
    for draw in 0..c.draws {
        if pool.is_empty() {
            pool.extend(FAMILIES);
        }
        let index = rng.index(pool.len());
        let family = pool.remove(index);
        for (task, classes) in [("digits", 10), ("charlm", 16)] {
            for &seed in &c.initialization_seeds {
                let source =
                    fs::read_to_string(root.join("models").join(family).join("model.wgt"))?
                        .replace("f32[3]", &format!("f32[{classes}]"))
                        .replace("f32[3,", &format!("f32[{classes},"));
                result.push(Trial {
                    id: format!("{task}-draw-{draw:02}-{family}-seed-{seed}"),
                    task: task.into(),
                    family: family.into(),
                    seed,
                    definition: ModelDefinition::parse(&source)?,
                });
            }
        }
    }
    Ok(result)
}
fn fans(family: &str, name: &str, shape: &[usize]) -> (f64, f64) {
    let receptive: usize = shape.iter().skip(2).product();
    if family == "transformer" && matches!(name, "ff1_weight" | "ff2_weight") {
        (shape[0] as f64, shape[1] as f64) // WGT matmul stores [in,out].
    } else {
        ((shape[1] * receptive) as f64, (shape[0] * receptive) as f64)
    }
}
fn initialized(t: &Trial, method: &str) -> Result<Weights> {
    ensure!(BASELINES.contains(&method), "unknown initializer");
    let mut weights = BTreeMap::new();
    for p in t.definition.parameters() {
        let mut rng = stream(t.seed, &p.name);
        let count = p.shape.iter().product();
        let values = if p.name.ends_with("_gamma") {
            vec![1.0; count]
        } else if p.shape.len() == 1 {
            vec![0.0; count]
        } else {
            let (fan_in, fan_out) = fans(&t.family, &p.name, &p.shape);
            (0..count)
                .map(|_| match method {
                    "normal" => (rng.normal() * 0.05) as f32,
                    "xavier_uniform" => {
                        ((2.0 * rng.unit() - 1.0) * (6.0 / (fan_in + fan_out)).sqrt()) as f32
                    }
                    _ => (rng.normal() * (2.0 / fan_in).sqrt()) as f32,
                })
                .collect()
        };
        weights.insert(p.name.clone(), values);
    }
    Ok(weights)
}
fn snapshot(t: &Trial, w: &Weights) -> Result<WeightSet> {
    ensure!(
        w.len() == t.definition.parameters().len(),
        "parameter key count mismatch"
    );
    let mut tensors = BTreeMap::new();
    for p in t.definition.parameters() {
        let values = w
            .get(&p.name)
            .with_context(|| format!("missing {}", p.name))?;
        tensors.insert(
            p.name.clone(),
            Tensor::new(p.shape.clone(), values.clone())?,
        );
    }
    let set = WeightSet::new(tensors)?;
    set.validate_for(&t.definition)?;
    Ok(set)
}
fn submission(path: &Path, trials: &[Trial]) -> Result<Submission> {
    let meta = fs::symlink_metadata(path)?;
    ensure!(
        meta.is_file() && meta.len() <= 8 * 1024 * 1024,
        "submission must be a regular JSON file <=8 MiB"
    );
    let s: Submission = serde_json::from_slice(&fs::read(path)?)?;
    ensure!(
        s.version == 1 && s.trials.len() == trials.len(),
        "submission version or trial count mismatch"
    );
    for t in trials {
        snapshot(t, s.trials.get(&t.id).context("missing trial")?)?;
    }
    Ok(s)
}

#[derive(Deserialize)]
struct Data {
    inputs: Vec<Vec<f32>>,
    labels: Vec<usize>,
}
fn dataset(
    root: &Path,
    task: &str,
    split: &str,
    classes: usize,
) -> Result<(Vec<Vec<f32>>, Vec<usize>)> {
    let d: Data = serde_json::from_slice(&fs::read(
        root.join("data").join(format!("{task}-{split}.json")),
    )?)?;
    ensure!(
        !d.inputs.is_empty() && d.inputs.len() == d.labels.len(),
        "invalid dataset lengths"
    );
    ensure!(
        d.inputs
            .iter()
            .all(|x| x.len() == 64 && x.iter().all(|v| v.is_finite()))
            && d.labels.iter().all(|&x| x < classes),
        "invalid data"
    );
    Ok((d.inputs, d.labels))
}
struct Service {
    child: Child,
    dir: tempfile::TempDir,
}
impl Service {
    fn start() -> Result<(Self, Client)> {
        let dir = tempfile::Builder::new().prefix("weight-init-").tempdir()?;
        fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o700))?;
        let daemon = std::env::var_os("WGTIME_DAEMON")
            .map(PathBuf::from)
            .unwrap_or(std::env::current_exe()?.with_file_name("wgtime"));
        let child = Command::new(daemon)
            .arg("--test-directory")
            .arg(dir.path())
            .env("WGTIME_COMPILE_CACHE", dir.path().join("compiled"))
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()?;
        let mut service = Self { child, dir };
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            ensure!(
                service.child.try_wait()?.is_none(),
                "wgtime exited during startup"
            );
            if service.dir.path().join("service.sock").exists() {
                if let Ok(client) = Client::connect_to(&service.dir.path().join("service.sock")) {
                    return Ok((service, client));
                }
            }
            ensure!(Instant::now() < deadline, "wgtime startup timeout");
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}
impl Drop for Service {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
fn metrics(model: &wgt::Model, data: &(Vec<Vec<f32>>, Vec<usize>)) -> Result<Value> {
    let mut correct = 0;
    let mut ce = 0.0f64;
    for (x, &label) in data.0.iter().zip(&data.1) {
        let logits = model.forward(x)?;
        let predicted = (0..logits.len())
            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]).then_with(|| b.cmp(&a)))
            .unwrap();
        correct += usize::from(predicted == label);
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        ce += max
            + logits
                .iter()
                .map(|&x| (x as f64 - max).exp())
                .sum::<f64>()
                .ln()
            - logits[label] as f64;
    }
    Ok(
        json!({"accuracy": correct as f64 / data.0.len() as f64, "cross_entropy": ce / data.0.len() as f64, "perplexity": (ce / data.0.len() as f64).exp().min(f64::MAX)}),
    )
}
fn evaluate(root: &Path, client: &Client, t: &Trial, w: &Weights, c: &Config) -> Result<Value> {
    let scratch = tempfile::tempdir()?;
    let directory = scratch.path().join("weights");
    let set = snapshot(t, w)?;
    set.save(&directory)?;
    let model = client.load_snapshot(
        &t.definition,
        &directory,
        Some(set.revision()),
        InstanceOptions {
            optimizer: Some(Optimizer::Adam),
            ..Default::default()
        },
    )?;
    ensure!(model.backend() == "native-cpu", "unexpected backend");
    let classes = t.definition.interface().unwrap().output;
    let train = dataset(root, &t.task, "train", classes)?;
    let test = dataset(root, &t.task, "test", classes)?;
    ensure!(
        train.0.len() == c.train_samples && test.0.len() == c.test_samples,
        "dataset/config count mismatch"
    );
    let initial = metrics(&model, &test)?;
    let mut rng = stream(c.train_seed, "minibatches");
    let mut losses = Vec::new();
    for _ in 0..c.steps {
        let indices: Vec<_> = (0..c.batch_size)
            .map(|_| rng.index(c.train_samples))
            .collect();
        let x: Vec<_> = indices.iter().map(|&i| train.0[i].clone()).collect();
        let y: Vec<_> = indices.iter().map(|&i| train.1[i]).collect();
        losses.push(model.train(
            &x,
            Targets::Classes(&y),
            TrainOptions {
                rate: c.learning_rate,
                loss: Loss::CrossEntropy,
                clip: Some(1.0),
            },
        )?);
    }
    let final_metrics = metrics(&model, &test)?;
    model.close()?;
    Ok(
        json!({"initial":initial, "final":final_metrics, "training_loss":losses, "weights_revision":set.revision()}),
    )
}
fn manifest(root: &Path, c: &Config, ts: &[Trial]) -> Result<Value> {
    let mut hashes = BTreeMap::new();
    for family in FAMILIES {
        hashes.insert(
            family,
            format!(
                "{:x}",
                Sha256::digest(fs::read(
                    root.join("models").join(family).join("model.wgt")
                )?)
            ),
        );
    }
    let mut data_hashes = BTreeMap::new();
    for task in ["digits", "charlm"] {
        for split in ["train", "test"] {
            let name = format!("{task}-{split}.json");
            let path = root.join("data").join(&name);
            if path.exists() {
                data_hashes.insert(name, format!("{:x}", Sha256::digest(fs::read(path)?)));
            }
        }
    }
    Ok(
        json!({"version":1, "weightlang_revision":REVISION.trim(), "config":c, "model_sha256":hashes, "data_sha256":data_hashes,
        "trials":ts.iter().map(|t| json!({"id":t.id,"task":t.task,"family":t.family,"seed":t.seed,
            "parameters":t.definition.parameters().iter().map(|p| json!({"name":p.name,"shape":p.shape})).collect::<Vec<_>>() })).collect::<Vec<_>>() }),
    )
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() >= 3,
        "usage: weight-init-bench manifest ROOT | emit ROOT METHOD | evaluate ROOT SUBMISSION REPORT | baselines ROOT REPORT"
    );
    let root = Path::new(&args[2]);
    let c = config(root)?;
    let ts = trials(root, &c)?;
    match args[1].as_str() {
        "manifest" => println!(
            "{}",
            serde_json::to_string_pretty(&manifest(root, &c, &ts)?)?
        ),
        "emit" => {
            let method = args.get(3).context("missing initializer")?;
            let s = Submission {
                version: 1,
                trials: ts
                    .iter()
                    .map(|t| Ok((t.id.clone(), initialized(t, method)?)))
                    .collect::<Result<_>>()?,
            };
            println!("{}", serde_json::to_string(&s)?);
        }
        "evaluate" | "baselines" => {
            let candidate = if args[1] == "evaluate" {
                Some(submission(
                    Path::new(args.get(3).context("missing submission")?),
                    &ts,
                )?)
            } else {
                None
            };
            let out = args
                .get(if candidate.is_some() { 4 } else { 3 })
                .context("missing report path")?;
            ensure!(
                !Path::new(out).exists(),
                "report exists; choose a new output path"
            );
            let (_service, client) = Service::start()?;
            let mut rows = Vec::new();
            let mut scores: BTreeMap<String, Vec<f64>> = BTreeMap::new();
            for t in &ts {
                eprintln!("evaluating {}", t.id);
                let mut methods = BASELINES.to_vec();
                if candidate.is_some() {
                    methods.push("candidate");
                }
                for method in methods {
                    let weights = if method == "candidate" {
                        candidate.as_ref().unwrap().trials[&t.id].clone()
                    } else {
                        initialized(t, method)?
                    };
                    let metrics = evaluate(root, &client, t, &weights, &c)
                        .with_context(|| format!("{} / {method}", t.id))?;
                    scores
                        .entry(method.into())
                        .or_default()
                        .push(metrics["final"]["accuracy"].as_f64().unwrap());
                    rows.push(json!({"trial":t.id,"task":t.task,"family":t.family,"method":method,"metrics":metrics}));
                }
            }
            let mut summaries = BTreeMap::new();
            let mut by_task = BTreeMap::new();
            for task in ["digits", "charlm"] {
                let mut summary = BTreeMap::new();
                for method in scores.keys() {
                    let subset: Vec<_> = rows
                        .iter()
                        .filter(|r| r["task"] == task && r["method"] == method.as_str())
                        .collect();
                    let mean = |key: &str| {
                        subset
                            .iter()
                            .map(|r| r["metrics"]["final"][key].as_f64().unwrap())
                            .sum::<f64>()
                            / subset.len() as f64
                    };
                    summary.insert(method, json!({"accuracy":mean("accuracy"),"cross_entropy":mean("cross_entropy"),"perplexity_from_mean_cross_entropy":mean("cross_entropy").exp().min(f64::MAX),"n":subset.len()}));
                }
                by_task.insert(task, summary);
            }
            for (method, values) in &scores {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let sd = if values.len() > 1 {
                    (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                        / (values.len() - 1) as f64)
                        .sqrt()
                } else {
                    0.0
                };
                summaries.insert(
                    method,
                    json!({"mean_final_accuracy":mean,"stddev_across_trials":sd,"n":values.len()}),
                );
            }
            let mut paired = BTreeMap::new();
            if let Some(values) = scores.get("candidate") {
                for method in BASELINES {
                    paired.insert(
                        method,
                        values
                            .iter()
                            .zip(&scores[method])
                            .map(|(a, b)| a - b)
                            .sum::<f64>()
                            / values.len() as f64,
                    );
                }
            }
            let reward = scores
                .get("candidate")
                .map(|v| v.iter().sum::<f64>() / v.len() as f64);
            let report = json!({"protocol":manifest(root,&c,&ts)?,"backend":"native-cpu","summary":summaries,"by_task":by_task,"paired_accuracy_delta":paired,"reward":reward,"trials":rows});
            fs::write(out, serde_json::to_vec_pretty(&report)?)?;
        }
        _ => bail!("unknown command"),
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e:#}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn root() -> &'static Path {
        Path::new(env!("CARGO_MANIFEST_DIR"))
    }
    #[test]
    fn models_and_sampling() {
        let c = config(root()).unwrap();
        let a = trials(root(), &c).unwrap();
        let b = trials(root(), &c).unwrap();
        assert_eq!(
            a.iter().map(|t| &t.id).collect::<Vec<_>>(),
            b.iter().map(|t| &t.id).collect::<Vec<_>>()
        );
        for family in FAMILIES {
            assert!(a.iter().any(|t| t.family == family));
        }
        for t in a {
            let i = t.definition.interface().unwrap();
            assert_eq!(i.input, 64);
            assert_eq!(i.output, if t.task == "digits" { 10 } else { 16 });
            for method in BASELINES {
                snapshot(&t, &initialized(&t, method).unwrap()).unwrap();
            }
        }
    }
    #[test]
    fn invalid_weights_are_rejected() {
        let c = config(root()).unwrap();
        let ts = trials(root(), &c).unwrap();
        let t = &ts[0];
        let mut w = initialized(t, "normal").unwrap();
        let key = w.keys().next().unwrap().clone();
        w.get_mut(&key).unwrap()[0] = f32::NAN;
        assert!(snapshot(t, &w).is_err());
        w.remove(&key);
        assert!(snapshot(t, &w).is_err());
        let mut w = initialized(t, "normal").unwrap();
        w.get_mut(&key).unwrap().push(0.0);
        assert!(snapshot(t, &w).is_err());
    }
    #[test]
    fn data_and_rng_are_reproducible_and_split() {
        let train = dataset(root(), "digits", "train", 10).unwrap();
        let test = dataset(root(), "digits", "test", 10).unwrap();
        assert_ne!(train, test);
        assert!(train.0.iter().all(|x| !test.0.contains(x)));
        assert_eq!(Rng(0).next(), 0xe220a8397b1dcdaf);
    }
    #[test]
    fn correct_fans_for_convolution_and_matmul() {
        assert_eq!(fans("cnn", "conv_weight", &[4, 1, 3, 3]), (9.0, 36.0));
        assert_eq!(fans("transformer", "ff1_weight", &[8, 16]), (8.0, 16.0));
        assert_eq!(fans("mlp", "hidden_weight", &[16, 64]), (64.0, 16.0));
    }
}
