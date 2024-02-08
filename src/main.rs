use peroxide::fuga::*;
use std::io::BufWriter;
use std::fs::File;

const ME: f64 = 0.5109989461; // MeV

#[allow(non_snake_case)]
fn main() {
    let parquet_file = "edata.parquet";
    // If parquet_file does not exist, convert csv to parquet
    if !std::path::Path::new(parquet_file).exists() {
        let mut df = DataFrame::read_csv("edata_modified.csv",',').unwrap();
        df.as_types(vec![F64, F64, F64, F64]);
        df.write_parquet(parquet_file, CompressionOptions::Uncompressed).unwrap();
    }


    let cs_file = "cs.bin";

    if !std::path::Path::new(cs_file).exists() {
        // Load electron stopping power data
        let df = DataFrame::read_parquet(parquet_file).unwrap();
        let E_k: Vec<f64> = df["E"].to_vec();
        let rad_stp_pow: Vec<f64> = df["rad_stp_pow"].to_vec();

        // Convert mass to liner
        let n_H = 0.1;
        let rad_stp_pow = rad_stp_pow.fmap(|s| convert_mass_to_linear(s, n_H));

        // Approximate collision stopping power for positron
        let E_tot = E_k.fmap(|E| (E.powi(2) + ME.powi(2)).sqrt());
        let apr_stp_pow = E_tot.fmap(|E| approx_linear_stopping_power(E, n_H));

        let tot_stp_pow = apr_stp_pow.add_v(&rad_stp_pow);

        // Cubic hermite spline for total stopping power
        let log_E_k = E_k.fmap(|E| E.log10());
        let cs = cubic_hermite_spline(&log_E_k, &tot_stp_pow, Quadratic);

        // Save cubic hermite spline using bincode
        let cs_file = "cs.bin";
        bincode::serialize_into(BufWriter::new(File::create(cs_file).unwrap()), &cs).unwrap();
    }

    let cs: CubicHermiteSpline = bincode::deserialize_from(File::open(cs_file).unwrap()).unwrap();
    let E_k = linspace(-2, 4, 1000);
    let stp_pow = cs.eval_vec(&E_k);
    let E_k = E_k.fmap(|E| 10f64.powf(E));

    let mut df = DataFrame::new(vec![]);
    df.push("E_k", Series::new(E_k));
    df.push("stp_pow", Series::new(stp_pow));
    df.print();

    df.write_parquet("stp_pow.parquet", CompressionOptions::Uncompressed).unwrap();
}

#[allow(non_snake_case)]
fn approx_linear_stopping_power(E: f64, n_H: f64) -> f64 {
    let gamma = E / ME;
    let beta2  = 1f64 - 1f64 / gamma.powi(2);
    n_H / 0.1 * 7.6e-26 / beta2 * (gamma.ln() + 6.6)
}

// n_H: number density of neutral hydrogen (cm^-3)
#[allow(non_snake_case)]
fn convert_mass_to_linear(s: f64, n_H: f64) -> f64 {
    s * n_H * 1.6735575e-24
}
