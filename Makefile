all:
	wasm-pack build --target web
debug:
	wasm-pack build --debug --target web

run:
	python3 -m http.server
