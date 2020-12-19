# musweeper
AI project using MuZero to play Minesweeper for [IMT3104](https://www.ntnu.no/studier/emner/IMT3104#tab=omEmnet)


## todo
- [x] Get muzero up and running (with tests)
- [x] Merge in [Gym-minesweeper](https://github.com/Zikoat/gym-minesweeper)
- [x] Add domain-specific knowledge to the observation space
- [x] Do classical steps if they are optimal
- [x] Use the domain-specific knowledge while training muzero (probability matrix form mrgris)
- [ ] Create a small test which trains the model and plays minesweeper, and calculates the average win rate.
- [ ] Create a simple script to run this in the colab notebook for offloading the training and eventual data analysis to Colab.
- [ ] Calculate average win rate on muzero
- [ ] Evaluate muzero vs [mrgris](http://mrgris.com/projects/minesweepr/)

## Install 
```
git submodule update --init --recursive # fetch submodules
cd gym-minesweeper && python setup.py install # install gym-enviorment
cd musweeper && python setup.py install # install musweeper
```

## Testing
```shell script
cd muzero
python -m unittest
```
Using pycharm: right-click test folder -> run 'unittest in test'

## Training a model
To train the model run `muzero_with_minesweeper.py` in the `examples` directory.  There is also code here to evaluate the trained model namely `evaluation_of_agents.py`. 

