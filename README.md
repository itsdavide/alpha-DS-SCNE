# alpha-DS-SCNE

Optimization code for the paper:
    
S. Lorenzini, D. Petturiti, B. Vantaggi. _Stackelberg-Cournot-Nash equilibria with Dempster-Shafer uncertainty and α-maxmin preferences._ 2026 

# Requirements
The code has been tested on Python 3.10 with the following libraries:
* **matplotlib** 3.7.1
* **numpy** 1.26.4
* **pandas** 1.5.3
* **pyomo** 6.6.1

Reference to the library **pyomo** is here: http://www.pyomo.org/

The code needs the **bonmin** solver that can be downloaded here: https://www.coin-or.org/Bonmin/

The **bonmin** solver should be located in a folder and the path to that folder should be inserted in the variable **optimizer_path** in the followin files:
* **Example_1_2_prox_E.py**
* **Example_1_2_prox_k.py**
* **Example_3_prox_E.py**
* **Example_3_prox_k.py**
* **Example_4_prox_E.py**
* **Example_4_prox_k.py**
