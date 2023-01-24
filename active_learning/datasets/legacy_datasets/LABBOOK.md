
## Questions?

* Validate Bork and Banks against our own lysates.
  * Sample compositions from both datasets and run them in house and compare results against reported yields to get R2.



## Initial investigations of Borkowski and Banks Datasets



## CFE Prep

|               | Borkowski      | Banks  |
| ------------- |:-------------:| -----:|
| Strain        | BL21 | BL21 DE3 Rosetta 2 |
| Protocol     | Sun 2013     |   Sun 2013 |
| OD600 @ Harvest      | 1.5      |   **2.0** |
| Lysis Method      | Sonnication      |   Sonnication |
| Dialysis | Yes      |    Yes |



## Reaction Conditions

|               | Borkowski      | Banks  |
| ------------- |:-------------:| -----:|
| Volume (ul)     | 10.5 | **100** |
|   Temp (C)    |   30  | 37 |




## Yield Definitions:

  * Bork "Yield": 
  
    > "The yield is defined as the ratio of the fluorescence produced with a chosen composition divided by the fluorescence obtained with the reference composition"

    Therefore it is the fold improvement.

    **I have asked Olivier to provide the raw data files so the delta FEU can be generated: Olivier hasn't responded. Will contact Jean-Loup.**

  * Banks "Yield": delta FEU between the peak RFU and the RFU at t = 0 for the same composition, blank subtracted.

## Jobs

* Compile combined tidy dataset will all possible metadata
  * Check end timepoints. Banks maybe inflection or total yield after elasped time
* Pretain models:
  * ~Bork~
  * Banks

### Analyses

* Sensitivity

