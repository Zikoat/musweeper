
clean:
	find . -name '*.pyc' -exec rm {} +
	find . -name '*.pyo' -exec rm {} +
	rm -rf ./build/
	rm -rf ./dist/
	rm -rf *.egg-info

basic_env:
	cd musweeper && python3 setup.py install
	cd examples && python3 muzero_with_basic_env.py