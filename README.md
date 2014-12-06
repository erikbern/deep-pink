deep-pink
=========

Deep Pink is a chess AI that learns to play chess using deep learning. [Here](http://erikbern.com/?p=841) is a  blog post providing some details about how it works.

There is a pre-trained model in the repo, but if you want to train your own model you need to download pgn files and run `parse_game.py`. After that, you need to run `train.py`, preferrably on a GPU machine since it will be 10-100x faster. This might take several days for a big model.

Note that the code is a bit hacky (eg. hardcoded paths in some places) so you might have to modify those to suit your needs.

Dependencies
============

* [Theano](https://github.com/Theano/Theano): `git clone https://github.com/Theano/Theano; cd Theano; python setup.py install`
* [Sunfish](https://github.com/thomasahle/sunfish): `git clone https://github.com/thomasahle/sunfish`. You need to add it to PYTHONPATH to be able to play
* [python-chess](https://pypi.python.org/pypi/python-chess) `pip install python-chess`
* [scikit-learn](http://scikit-learn.org/stable/install.html) (only needed for training)
* [h5py](http://www.h5py.org/): can be installed using `apt-get install python-hdf5` or `pip install hdf5` (only needed for training)
