use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{loss, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use std::time::Instant;

fn generate_data(device: &Device) -> Result<(Tensor, Tensor)> {
    let n = 100;
    let x: Vec<f32> = (0..n).map(|x| x as f32 * 0.2 - 10.0).collect();
    let y: Vec<f32> = x
        .iter()
        .map(|&xi| 2.0 * xi.powi(3) - 5.0 * xi.powi(2) + 10.0 * xi - 7.0)
        .collect();

    let x_tensor = Tensor::from_vec(x, (n, 1), device)?;
    let y_tensor = Tensor::from_vec(y, (n, 1), device)?;
    Ok((x_tensor, y_tensor))
}

#[derive(Debug)]
struct RegressionModel {
    fc1: candle_nn::Linear,
    fc2: candle_nn::Linear,
    fc3: candle_nn::Linear,
}

impl RegressionModel {
    fn new(vb: VarBuilder) -> Result<Self> {
        let fc1 = candle_nn::linear(1, 64, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(64, 64, vb.pp("fc2"))?;
        let fc3 = candle_nn::linear(64, 1, vb.pp("fc3"))?;
        Ok(Self { fc1, fc2, fc3 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?.relu()?;
        let xs = self.fc2.forward(&xs)?.relu()?;
        self.fc3.forward(&xs)
    }
}

fn main() -> Result<()> {
    let start = Instant::now();
    let device = Device::Cpu;
    let (x_train, y_train) = generate_data(&device)?;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = RegressionModel::new(vs)?;
    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        },
    )?;

    for epoch in 0..5000 {
        let preds = model.forward(&x_train)?;
        let loss = loss::mse(&preds, &y_train)?;
        opt.backward_step(&loss)?;

        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {:?}", epoch, loss.to_scalar::<f32>()?);
        }
    }

    let duration = start.elapsed();
    println!("Time taken: {:?}", duration);
    Ok(())
}
