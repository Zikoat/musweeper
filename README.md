# musweeper
AI project using MuZero to play Minesweeper for [IMT3104](https://www.ntnu.no/studier/emner/IMT3104#tab=omEmnet)


## todo
- [ ] Get muzero up and running (with tests)
- [x] Merge in [Gym-minesweeper](https://github.com/Zikoat/gym-minesweeper)
- [ ] Evaluate muzero vs [mrgris](http://mrgris.com/projects/minesweepr/)
- [ ] Use mrgris for giving the model "dark Knowledge" (simulate states from mrgris and use his actions as optimal)
- [ ] Create a small test which trains the model and plays minesweeper, and calculates the average win rate.
- [ ] Create a simple script to run this in the colab notebook for offloading the training and eventual data analysis to Colab.
- [ ] Calculate average win rate on muzero

## Testing

```shell script
cd muzero
python -m unittest
```
Using pycharm: right-click test folder -> run 'unittest in test'
