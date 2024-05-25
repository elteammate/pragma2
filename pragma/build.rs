fn main() {
    println!("cargo:rerun-if-changed=tests/suite/**/*.pragma");
}
