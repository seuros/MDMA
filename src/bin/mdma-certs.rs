// Generates a self-signed CA plus a server cert and one client cert, both
// signed by that CA. Each invocation creates a new CA; its private key
// is not persisted, so this helper cannot issue additional clients for it.
//
// Usage: mdma-certs [output-dir]  (defaults to ./certs)
//        mdma-certs --spki <cert.pem>  prints the certificate's pin for MDMA_OPERATOR_SPKI
//
// Writes: ca.pem, server.pem, server-key.pem, client.pem, client-key.pem,
// the keys readable by their owner only. Never replaces existing files.
//
// Env vars mdma-serve reads from this output:
//   MDMA_TLS_CERT=<dir>/server.pem
//   MDMA_TLS_KEY=<dir>/server-key.pem
//   MDMA_TLS_CA=<dir>/ca.pem
use rama::error::BoxError;
use rcgen::{
    BasicConstraints, CertificateParams, DistinguishedName, DnType, ExtendedKeyUsagePurpose, IsCa,
    Issuer, KeyPair,
};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

const FILES: [&str; 5] = [
    "ca.pem",
    "server.pem",
    "server-key.pem",
    "client.pem",
    "client-key.pem",
];

/// Creates `path` holding `pem`; a `private` file is readable by its owner only.
fn create(path: &Path, pem: &str, private: bool) -> Result<(), BoxError> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    if private {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    file.write_all(pem.as_bytes())?;
    file.sync_all()?;
    Ok(())
}

fn write_pair(dir: &Path, name: &str, cert_pem: &str, key_pem: &str) -> Result<(), BoxError> {
    create(&dir.join(format!("{name}.pem")), cert_pem, false)?;
    create(&dir.join(format!("{name}-key.pem")), key_pem, true)
}

fn ca_params() -> Result<CertificateParams, BoxError> {
    let mut params = CertificateParams::new(Vec::<String>::new())?;
    params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    let mut dn = DistinguishedName::new();
    dn.push(DnType::CommonName, "mdma federation CA");
    params.distinguished_name = dn;
    Ok(params)
}

fn leaf_params(
    common_name: &str,
    san: Vec<String>,
    client_auth: bool,
) -> Result<CertificateParams, BoxError> {
    let mut params = CertificateParams::new(san)?;
    let mut dn = DistinguishedName::new();
    dn.push(DnType::CommonName, common_name);
    params.distinguished_name = dn;
    if client_auth {
        params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ClientAuth];
    } else {
        params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
    }
    Ok(params)
}

fn main() -> Result<(), BoxError> {
    if std::env::args().nth(1).as_deref() == Some("--spki") {
        let path = std::env::args()
            .nth(2)
            .ok_or("usage: mdma-certs --spki <cert.pem>")?;
        println!(
            "{}",
            mdma::grpc::client::spki_pin(path.as_ref()).map_err(|e| format!("{path}: {e}"))?
        );
        return Ok(());
    }
    let out_dir: PathBuf = std::env::args()
        .nth(1)
        .map_or_else(|| PathBuf::from("certs"), PathBuf::from);
    fs::create_dir_all(&out_dir)?;
    // Checked up front so a refusal writes nothing: servers and clients may
    // still trust what is there.
    if let Some(existing) = FILES
        .iter()
        .map(|name| out_dir.join(name))
        .find(|path| fs::symlink_metadata(path).is_ok())
    {
        return Err(format!(
            "{} exists; mdma-certs never replaces credentials, so name an empty directory",
            existing.display()
        )
        .into());
    }

    let ca_key = KeyPair::generate()?;
    let ca_params = ca_params()?;
    let ca_cert = ca_params.self_signed(&ca_key)?;
    create(&out_dir.join("ca.pem"), &ca_cert.pem(), false)?;
    let issuer = Issuer::from_params(&ca_params, &ca_key);

    let server_key = KeyPair::generate()?;
    let server_params = leaf_params(
        "mdma-server",
        vec!["localhost".to_string(), "127.0.0.1".to_string()],
        false,
    )?;
    let server_cert = server_params.signed_by(&server_key, &issuer)?;
    write_pair(
        &out_dir,
        "server",
        &server_cert.pem(),
        &server_key.serialize_pem(),
    )?;

    let client_key = KeyPair::generate()?;
    let client_params = leaf_params("mdma-client", vec![], true)?;
    let client_cert = client_params.signed_by(&client_key, &issuer)?;
    write_pair(
        &out_dir,
        "client",
        &client_cert.pem(),
        &client_key.serialize_pem(),
    )?;

    eprintln!("wrote CA + server + client certs to {}", out_dir.display());
    eprintln!();
    eprintln!("export MDMA_TLS_CERT={}/server.pem", out_dir.display());
    eprintln!("export MDMA_TLS_KEY={}/server-key.pem", out_dir.display());
    eprintln!("export MDMA_TLS_CA={}/ca.pem", out_dir.display());
    eprintln!();
    eprintln!(
        "client.pem/client-key.pem are for a client connecting to this server (e.g. mdma-client),"
    );
    eprintln!(
        "signed by the same CA at ca.pem so the server's ClientAuth verification accepts them."
    );

    Ok(())
}
