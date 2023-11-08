
## Roadmap

#### quantificator domain

  - single multilabel quantificator
  
  - vector of binary quantificators
  
    | quantificator       |                |                |
    |:-------------------:|:--------------:|:--------------:|
    | true quantificator  | true positive  | false positive |
    | false quantificator | false negative | true negative  |

#### dataset split
  
  - train | test
    - classificator C is fit on train
    - quantificator Q is fit on cross validation of C over train
  - train | validation | test
    - classificator C is fit on train
    - quantificator Q is fit on validation
    
#### classificator origin

  - black box
  - crystal box

#### test metrics

  - f1_score
  - K

#### models

  - classificator
  - quantificator



