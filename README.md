# INFO7250-Final
Final Project of Engineering Big-Data Systems

Check these first:<br>
<a href="https://github.com/wadehuangwei/INFO7250-Final/blob/master/7250Final.pptx">Presentation</a><br>
<a href="https://github.com/wadehuangwei/INFO7250-Final/blob/master/Report.docx">Report</a>

Prerequisites:<br>
You need to have the following installed:<br>
python3<br>
Jupyter4.2.3<br>
Spark2.1, should be configured to be driven by jupyter. <a href="https://medium.com/@GalarnykMichael/install-spark-on-mac-pyspark-453f395f240b">Configuration guide</a><br>
The following python libraries are needed:<br>
Theano, Tensorflow, Keras

To run:

1. Install Git and clone this repo

2. Download all the content of <a href="https://drive.google.com/open?id=0B-cyNEbHVKHCczZ3RlAwaFQwWlE">this</a> folder and put them in INFO7250-Final/dataset

3. Navigate to INFO7250-Final/code folder in your terminal
```sh
$ cd path/to/INFO7250-Final/code
```

4. Data preprocessing. In jupyter, run "large data processing.ipynb" and "small data processing.ipynb"

5. Run ANN on small dataset. Check the output in terminal.
```sh
$ python ann_s.py
```

6. Run ANN on large dataset. Check the output in terminal.
```sh
$ python ann_l.py
```
