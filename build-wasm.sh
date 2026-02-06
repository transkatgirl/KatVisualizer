trunk build --release -M
wasm-opt -O4 -all dist/katvisualizer_wasm_bg.wasm -o dist/katvisualizer_wasm_bg.opt.wasm
rm dist/katvisualizer_wasm_bg.wasm
mv dist/katvisualizer_wasm_bg.opt.wasm dist/katvisualizer_wasm_bg.wasm