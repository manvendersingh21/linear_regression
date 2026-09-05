use std::fs;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};


fn main(){
    let content =fs::read_to_string("Housing.csv").expect("Unable to read file");
    let mut x: Vec<Vec<f64>> = Vec::new();
    let mut y: Vec<f64> = Vec::new();
    for line in content.lines().skip(1){
        let cols: Vec<&str> = line.split(',').collect();
        let price: f64 = cols[0].parse().unwrap();

        let mut row: Vec<f64> = vec![1.0];
        row.push(cols[1].parse::<f64>().unwrap()); // area
        row.push(cols[2].parse::<f64>().unwrap()); // bedrooms
        row.push(cols[3].parse::<f64>().unwrap()); // bathrooms
        row.push(cols[4].parse::<f64>().unwrap()); // stories
        row.push(if cols[5] == "yes" { 1.0 } else { 0.0 }); // mainroad
        row.push(if cols[6] == "yes" { 1.0 } else { 0.0 }); // guestroom
        row.push(if cols[7] == "yes" { 1.0 } else { 0.0 }); // basement
        row.push(if cols[8] == "yes" { 1.0 } else { 0.0 }); // hotwaterheating
        row.push(if cols[9] == "yes" { 1.0 } else { 0.0 }); // airconditioning
        row.push(cols[10].parse::<f64>().unwrap()); // parking
        row.push(if cols[11] == "yes" { 1.0 } else { 0.0 }); // prefarea
        row.push(match cols[12].trim() { // furnishingstatus
            "furnished" => 1.0,
            "semi-furnished" => 2.0,
            _ => 3.0,
        });

        x.push(row);
        y.push(price);

    }
    let mut rng = thread_rng();
    let mut indices: Vec<usize> = (0..x.len()).collect();
    indices.shuffle(&mut rng);
    let mut x: Vec<Vec<f64>> = indices.iter().map(|&i| x[i].clone()).collect();
    let y: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
    // Normalize x (skip bias at index 0)
    let num_features = x[0].len();
    for i in 1..num_features {
        let mean = x.iter().map(|row| row[i]).sum::<f64>() / x.len() as f64;
        let std = (x.iter().map(|row| (row[i] - mean).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
        let std = if std == 0.0 { 1.0 } else { std };
        for row in x.iter_mut() {
            row[i] = (row[i] - mean) / std;
        }
    }

    // Normalize y
    let y_mean = y.iter().sum::<f64>() / y.len() as f64;
    let y_std = (y.iter().map(|v| (v - y_mean).powi(2)).sum::<f64>() / y.len() as f64).sqrt();
    let  y: Vec<f64> = y.iter().map(|v| (v - y_mean) / y_std).collect();
    let split = (x.len() as f64 * 0.9) as usize;

    let x_train = &x[..split];
    let x_test = &x[split..];
    let y_train = &y[..split];
    let y_test = &y[split..];


    let mut theta = vec![0.0; x[0].len()];
    let alpha = 0.01;
    let iter = 5000;
    println!("🚀 Training Linear Regression Model (alpha: {}, iterations: {})...", alpha, iter);
    train(&mut theta, &x_train, &y_train, alpha, iter);
    println!("Trained parameters: {:?}", theta);
    let mut sse = 0.0;
    let mut sst = 0.0;
    let y_test_mean = y_test.iter().sum::<f64>() / y_test.len() as f64;

    for i in 0..x_test.len(){
        let pred = hypothesis(&theta, &x_test[i]);
        let actual = y_test[i];
        sse += (actual - pred).powi(2);
        sst += (actual - y_test_mean).powi(2);

        let pred_unscaled = pred * y_std + y_mean;
        let actual_unscaled = actual * y_std + y_mean;
        println!("Predicted: {:.2}, Actual: {:.2}", pred_unscaled, actual_unscaled);
    }

    let mse = sse / x_test.len() as f64;
    let r2 = 1.0 - (sse / sst);
    println!("\n=== Model Evaluation ===");
    println!("Normalized MSE : {:.6}", mse);
    println!("R^2 Score      : {:.4}", r2);
}

fn hypothesis(theta: &Vec<f64>, x: &[f64]) -> f64{
    let mut sum = 0.0;
    for i in 0..theta.len(){
        sum += theta[i] * x[i];
    }
    return sum;
}
fn train(theta: &mut Vec<f64>, x: &[Vec<f64>], y: &[f64], alpha: f64, iter: i32){
    for epoch in 0..iter{
        let mut temp = vec![0.0; theta.len()];
        let mut total_loss = 0.0;
        for i in 0..theta.len(){
            let mut sum = 0.0;
            for j in 0..x.len(){
                let diff = hypothesis(theta, &x[j]) - y[j];
                sum += diff * x[j][i];
                if i == 0 {
                    total_loss += diff.powi(2);
                }
            }
            temp[i] = theta[i] - alpha * sum / x.len() as f64;
        }
        for i in 0..theta.len(){
            theta[i] = temp[i];
        }
        if (epoch + 1) % 1000 == 0 || epoch == iter - 1 {
            let current_mse = total_loss / (2.0 * x.len() as f64);
            println!("  [Epoch {:>4}/{}] Training Loss (MSE): {:.6}", epoch + 1, iter, current_mse);
        }
    }
}
fn sgd_train(theta: &mut Vec<f64>, x: &[Vec<f64>], y: &[f64], alpha: f64, iter: i32){
    let mut rng = thread_rng();
    for _ in 0..iter{
        let j = rng.gen_range(0..x.len());
        for i in 0..theta.len(){
            theta[i] -= alpha * (hypothesis(theta, &x[j]) - y[j]) * x[j][i];
        }
    }
}

fn transpose(m: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows = m.len();
    let cols = m[0].len();
    let mut result = vec![vec![0.0; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            result[j][i] = m[i][j];
        }
    }
    result
}

fn mat_mul(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = b[0].len();
    let inner = b.len();
    let mut result = vec![vec![0.0; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            for k in 0..inner {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

fn mat_vec_mul(a: &Vec<Vec<f64>>, b: &Vec<f64>) -> Vec<f64> {
    let rows = a.len();
    let cols = a[0].len();
    let mut result = vec![0.0; rows];
    for i in 0..rows {
        for j in 0..cols {
            result[i] += a[i][j] * b[j];
        }
    }
    result
}

fn inverse(m: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = m.len();
    let mut aug = vec![vec![0.0; 2 * n]; n];

    // Build augmented matrix [m | I]
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = m[i][j];
        }
        aug[i][i + n] = 1.0;
    }

    // Gauss-Jordan elimination
    for col in 0..n {
        // Find pivot
        let mut pivot = col;
        for row in col+1..n {
            if aug[row][col].abs() > aug[pivot][col].abs() {
                pivot = row;
            }
        }
        aug.swap(col, pivot);

        let diag = aug[col][col];
        for j in 0..2*n {
            aug[col][j] /= diag;
        }

        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..2*n {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    // Extract right half
    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            result[i][j] = aug[i][j + n];
        }
    }
    result
}

fn normal_equation(x: &Vec<Vec<f64>>, y: &Vec<f64>) -> Vec<f64> {
    let xt = transpose(x);
    let xt_x = mat_mul(&xt, x);
    let xt_x_inv = inverse(&xt_x);
    let xt_y = mat_vec_mul(&xt, y);
    mat_vec_mul(&xt_x_inv, &xt_y)
}
