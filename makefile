line1="import sys"
line2="import os"
line3="print\(1\)" #"os.system\(\"pip install -r ../input/wuINSERTUGHERE/requirments.txt\"\)"
line4="os.system\(\"pip install ../input/wuINSERTUGHERE/musweeper-0.0.1-py2.py3-none-any.whl\"\)"

clean:
	find . -name '*.pyc' -exec rm {} +
	find . -name '*.pyo' -exec rm {} +
	rm -rf ./musweeper/build/
	rm -rf ./musweeper/dist/
	rm -rf *.egg-info

basic_env:
	cd musweeper && python3 setup.py install
	cd examples && python3 muzero_with_basic_env.py

run_on_kaggle:
	cd musweeper && python3.8 setup.py bdist_wheel --universal
	scp "./musweeper/dist/musweeper-0.0.1-py2.py3-none-any.whl" $$rebox:/root/kaggledataset
	send.sh "./" "cd /root/kaggledataset/ && kaggle datasets version -p ./ -m "$$(date)"" cmd	

	echo "$(line1)" > /Users/2xic/master/projects/common/shecdule/.tmpkaggle.py
	echo "$(line2)" >> /Users/2xic/master/projects/common/shecdule/.tmpkaggle.py
	echo "$(line3)" >> /Users/2xic/master/projects/common/shecdule/.tmpkaggle.py
	echo "$(line4)" >> /Users/2xic/master/projects/common/shecdule/.tmpkaggle.py

	cat "./examples/muzero_with_basic_env.py" >> /Users/2xic/master/projects/common/shecdule/.tmpkaggle.py
	cd /Users/2xic/master/projects/common/shecdule/ && python3 compiler.py kaggle /Users/2xic/master/projects/common/shecdule/.tmpkaggle.py
	python3 /Users/2xic/user_scripts/other-production/kaggle-offloader/kaggle_latest_response.py
