use nih_plug::prelude::*;

use katvisualizer::MyPlugin;

fn main() {
    nih_export_standalone::<MyPlugin>();
}
