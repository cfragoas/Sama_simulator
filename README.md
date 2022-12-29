# Sama*

Sama is a open-source, Python based simulator for 5G Radio Access Networks (RANs) that supports both downlink and uplink communications between spatially distributed User Equipments (UEs) and arbitrary Base Station (BS) arrangements, and considers the intrinsic variability of the wireless channel between UEs and BSs.

As a tool, it is capable of emulating diverse RAN scenarios, since it encompasses the necessary minutiae to conduct simulations under different systemic configurations and equipment characteristics with little user input. Sama can evaluate each simulation scenario and assess the requirements of the different use-cases, based on the Channel State Information (CSI, or, equivalently, the Channel Quality Indicator, CQI) between UEs and BSs.

The radio-link CSI from a BS to a UE is a proxy for the link’s capacity. The associations between UEs and the BS’s beams are obtained from the CSI, spatially portraying the RAN’s radio links among UEs and BSs, also considering the channel’s variability.

To evaluate the mobile network service, it simulates the multiple accessess of UEs to the Physical Layer (PL) frame resources considering the spatial arrangement of the RAN elements and its equipment characteristics. Subsequently, the RAN performance is evaluated as the scheduling of the time-frequency resources of a PL frame among the UEs takes place.

The simulator returns UEs–BSs-beams associations and the resulting links’ path losses and data rates, besides the UE capacity and latency. The result is a snap of the RAN during a PL frame. The RAN performance can be accessed using the Monte Carlo approach, considering many PL frames.
 
Sama also allows evolving the legacy infrastructure sites (cabinets and poles) to support a new RAN deployment.

Please refer to the papers for more information: [[1]], [[2]]

*Sama is the Tupi's word for rope/cable and is the acronym for Simulation and Analysis of Mobile Access.

### Libraries

Sama uses different open source python libraries, such as:
* [matplotlib] - Comprehensive library for creating static, animated, and interactive visualizations in Python
* [numba] - JIT compiler that translates a subset of Python and NumPy code into fast machine code
* [scikit-learn] - Simple and efficient tools for predictive data analysis
* [seaborn] - Python data visualization library that provides a high-level interface for drawing attractive and informative statistical graphics
* [pickle-mixin] - Makes un-pickle-able objects pick-able
* [tqdm] - Instantly make your loops show a smart progress meter
* [pandas] - Fast, powerful, flexible and easy to use open source data analysis and manipulation tool
* [numpy] - The fundamental package for scientific computing with Python
* [pyyaml] - Full-featured YAML framework for the Python programming language

### Installation

Sama is currently tested only in Python 3, using [miniforge].
Install Python, copy all folders and files to any directory and make sure all packages are installed.
To install the packages, run the following commands:

```sh
pip install -r requirements.txt
```

or

```sh
pip install matplotlib==3.6.2
pip install numba==0.56.4
pip install tqdm==4.64.1
pip install pandas==1.5.2
pip install scikit-learn==1.2.0
pip install seaborn==0.12.1
pip install pyyaml==0.2.5
pip install pickle-mixin==1.0.2
```

Setup the parameter file **param.yml** in the folder **parameters/**, linked [here]. Alter according to the simulations you choose to execute, but you must not change the name nor location of the file. 

Run:

```sh
python Run.py
```

to start Sama.

### Contributions

Since this is still a early version of the code, we expect that some problems can be found. We are tottaly open for contributions and bug/problems reports.


### Authorship

All the code development was made by Christian Rodrigues.
Contact: christianfragoas@gmail.com



[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

[miniforge]: <https://github.com/conda-forge/miniforge>
[tqdm]: <https://github.com/tqdm/tqdm>
[pandas]: <https://pandas.pydata.org/>
[numpy]: <https://numpy.org/>
[matplotlib]: <https://matplotlib.org/>
[numba]: <https://numba.pydata.org/>
[scikit-learn]: <https://scikit-learn.org/>
[seaborn]: <https://seaborn.pydata.org/>
[pickle-mixin]: <https://pypi.org/project/pickle-mixin/>
[pyyaml]: <https://pyyaml.org/>

[here]: <https://github.com/cfragoas/CelDep_Optimizator/blob/main/parameters/param.yml>
[1]: <http://dx.doi.org/10.14209/sbrt.2022.1570814168>
[2]: <>


[//]: # (# CelDep_Optimizator)

[//]: # (usando miniforge como auxílio)
[//]: # (https://github.com/conda-forge/miniforge)
