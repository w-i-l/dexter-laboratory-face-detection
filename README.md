
<br>
<hr>
<h2>How to use it</h2>

>**NOTE**: For this you will need to have <code>conda</code> installed. If you do not have it, you can install it by following the instructions from the <a href="https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html">official documentation</a>.

<p>For using the provided application you will need to install the dependecies by running the following code:</p>


```bash
conda create -n dexter python=3.10
conda activate dexter
conda install matplotlib numpy opencv scipy tqdm -y
pip install tf_keras
```

<p>For creating a Python virtual environment you can use the following <a href="https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/">guide</a>. Then install the dependencies using <code>pip</code>.</p>


<p>This will create a new environment called <code>dexter</code> and install the required packages, using the <code>Python3</code>.</p>

<p>After that we will want to install the <code>Tensorflow</code> package with the MPS capabilities for the <code>MacOS</code>. This will ensure a faster processing time. For this we will use the following code:</p>

```bash
conda install -c apple tensorflow-deps
pip install tensorflow-macos==2.15.0
pip install tensorflow-metal==1.1.0
```

For `Windows` you can use the following code:

```bash
pip install tensorflow
```

<p>For other platforms or distribution, please refer to <a href"https://www.tensorflow.org/install/pip">offical documentation</a>.</p>

<p>After the instalation is complete, you will need to create a folder in the <code>data</code> directory called <code>test</code>. Here you can put all of the images.

<p>So the folder structure should look like this:</p>

```
data
│
└───test
│   │
│   └───001.jpg
│   │   002.jpg
│   │   ...
```

<p>After that you can run the following code:</p>

```bash
python main.py
```

>**Note:** If you encounter the following error: <code>ImportError: attempted relative import with no known parent package</code>  or <code>ModuleNotFoundError: No module named</code> make sure to run the script from the <code>src</code> folder and if that does not work, try to run the script with the following code: 

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

<p>This will generate the predictions for each task in the <code>data/output</code> directory.</p>

<p>On my machine the processing time for 200 images with the <code>MPS</code> capabilities enabled was around 13 minutes. Also keep in mind the data loaders for the CNN models are using a small batches to ensure a lower use of hardware resources.</p>
