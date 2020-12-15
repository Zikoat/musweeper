line1="import sys"
line2="import os"
line3="os.system\(\"pip install -r ../input/wuINSERTUGHERE/requirments.txt\"\)"
line4="os.system\(\"pip install ../input/wuINSERTUGHERE/musweeper-0.0.1-py2.py3-none-any.whl\"\)"
line5="os.system\(\"pip install ../input/wuINSERTUGHERE/hydra-0.0.2-py2.py3-none-any.whl\"\)"
line6="os.system\(\"pip install ../input/wuINSERTUGHERE/gym_minesweeper-0.0.1-py3-none-any.whl\"\)"

clean:
	find . -name '*.pyc' -exec rm {} +
	find . -name '*.pyo' -exec rm {} +
	rm -rf ./musweeper/build/
	rm -rf ./musweeper/dist/
	rm -rf *.egg-info


build:
	cd musweeper && python3 setup.py install

eval: build
	cd examples && python3 evaluat_models.py

run_example: build
	cd examples && python3 muzero_with_basic_env.py

run_example2: build
	cd examples && python3 muzero_with_minesweeper.py

run_example3: build
	cd examples && python3 muzero_with_cartpole.py

debug: build
#	cd examples && python3 visualizations.py
#	cd examples && python3 debug_trained_model.py
	cd examples && python3 muzero_with_minesweeper_v1.py


run_on_kaggle:
	cd musweeper && python3.8 setup.py bdist_wheel --universal
	scp "./musweeper/dist/musweeper-0.0.1-py2.py3-none-any.whl" $$rebox:/root/kaggledataset
	send.sh "./" "cd /root/kaggledataset/ && kaggle datasets version -p ./ --dir-mode zip -m \"$$(date)\"" cmd	

	echo "$(line1)" > /Users/2xic/master/projects/common/shecdule/.tmpkaggle.py
	echo "$(line2)" >> /Users/2xic/master/projects/common/shecdule/.tmpkaggle.py
	echo "$(line3)" >> /Users/2xic/master/projects/common/shecdule/.tmpkaggle.py
	echo "$(line4)" >> /Users/2xic/master/projects/common/shecdule/.tmpkaggle.py
	echo "$(line5)" >> /Users/2xic/master/projects/common/shecdule/.tmpkaggle.py
	echo "$(line6)" >> /Users/2xic/master/projects/common/shecdule/.tmpkaggle.py

#	cat "./examples/muzero_with_basic_env.py" >> /Users/2xic/master/projects/common/shecdule/.tmpkaggle.py
	cat "./examples/muzero_with_minesweeper.py" >> /Users/2xic/master/projects/common/shecdule/.tmpkaggle.py
	cd /Users/2xic/master/projects/common/shecdule/ && python3 compiler.py kaggle /Users/2xic/master/projects/common/shecdule/.tmpkaggle.py
#	python3 /Users/2xic/user_scripts/other-production/kaggle-offloader/kaggle_latest_response.py

run_remote:
	echo "building ..."
	send.sh "./" norun
	send.sh "./README.md" norun
	send.sh "./" "cd musweeper && python3.8 setup.py install" cmd
	send.sh "./" "cd examples && python3.8 muzero_with_basic_env.py" cmd

cov:
	coverage run -m pytest
	coverage report -m

overleaf:
	git submodule foreach git pull origin master
	./converter.sh

